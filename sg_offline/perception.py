"""Causal perception adapters for controlled RGB-D recordings. No reference labels at runtime."""
from dataclasses import dataclass, field, replace
from itertools import permutations
import numpy as np
from .contracts import Packet, SIDES, Stamp
from .ledger import FUNCTIONAL_POINTS


@dataclass(frozen=True)
class ObjectObservation:
    """One object as a provider observed it in one frame.

    ``xyz`` is a metric point in the frame's optical frame computed from the registered depth (a region
    centroid), or None with a ``reason``. ``pixel`` is a representative image location for diagnostics.
    ``details`` carries provider measurements (height above the table, footprint extent, point count, mask
    area, source) that go to the perception diagnostics, never into the packet."""
    identity: str
    xyz: tuple | None
    pixel: tuple | None = None
    reason: str = ""
    definition: str = "object_region_centroid_above_table_plane"
    details: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.identity or self.identity in SIDES:
            raise ValueError("object identity required and must not be a hand name")
        if self.xyz is None and not self.reason:
            raise ValueError("an unavailable object needs a reason")
        if self.xyz is not None and (len(self.xyz) != 3 or not np.isfinite(self.xyz).all()):
            raise ValueError("object xyz must be finite")


def box_iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = float(wh.prod())
    union = float(np.maximum(a[2:]-a[:2], 0).prod() + np.maximum(b[2:]-b[:2], 0).prod() - intersection)
    return intersection / union if union else 0.


def box_overlap_of_smaller(a, b):
    """Intersection divided by the SMALLER box's area: 1.0 when one box lies inside the other."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    smaller = min(float(np.maximum(a[2:]-a[:2], 0).prod()), float(np.maximum(b[2:]-b[:2], 0).prod()))
    return float(wh.prod()) / smaller if smaller else 0.


def _nearest_valid_pixel(depth, mask, centre, invalid=0):
    """(column, row) inside the mask with valid depth nearest to ``centre``; None if there is none.
    Same rule as ``bench_segments.nearest_valid_pixel``; kept here so the runtime package imports no script."""
    # the same validity test as RGBDFrame.point_at: the marker, non-finite values and non-positive depth are all invalid
    depth = np.asarray(depth)
    valid = (depth != invalid) & (depth > 0) & (np.isfinite(depth) if depth.dtype.kind == "f" else True)
    rows, cols = np.nonzero(mask & valid)
    if rows.size == 0:
        return None
    k = int(np.argmin((cols - centre[0]) ** 2 + (rows - centre[1]) ** 2))
    return int(cols[k]), int(rows[k])


def boxes_touch(a, b):
    """True when two boxes share any area (threshold-free geometric test)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool(np.all(np.minimum(a[2:], b[2:]) > np.maximum(a[:2], b[:2])))


def match_boxes(predictions, references, minimum_iou=.3):
    """Evaluation only: one-to-one maximum-IoU matching, independent of handedness."""
    from itertools import combinations
    best, best_score = [], -1.
    for n in range(min(len(predictions), len(references)) + 1):
        for selected in combinations(range(len(predictions)), n):
            for targets in permutations(range(len(references)), n):
                pairs = list(zip(selected, targets))
                scores = [box_iou(predictions[i], references[j]) for i, j in pairs]
                if all(v >= minimum_iou for v in scores) and n + sum(scores)/(n+1) > best_score:
                    best, best_score = pairs, n + sum(scores)/(n+1)
    return best


@dataclass(frozen=True)
class HandDetection:
    box: tuple[float, float, float, float]
    side: str | None
    confidence: float
    points: dict  # three names -> (u,v) or None; never fill missing points with a different definition
    model: str
    definition: str = "three_functional_points"
    extra_points: dict = field(default_factory=dict)  # EXTRA_LANDMARK_NAMES -> (u,v) or None (fingertips); optional

    def __post_init__(self):
        if len(self.box) != 4 or not np.isfinite(self.box).all() or self.box[2] <= self.box[0] or self.box[3] <= self.box[1]:
            raise ValueError("finite ordered xyxy box required")
        if self.side not in (*SIDES, None) or not 0 <= self.confidence <= 1:
            raise ValueError("invalid detection identity or confidence")
        if set(self.points) != set(FUNCTIONAL_POINTS) or not self.model:
            raise ValueError("explicit three functional point predictions and model required")
        from .scene import EXTRA_LANDMARK_NAMES
        if not set(self.extra_points) <= set(EXTRA_LANDMARK_NAMES):
            raise ValueError(f"extra points must be among {EXTRA_LANDMARK_NAMES}")
        for pixel in list(self.points.values()) + list(self.extra_points.values()):
            if pixel is not None and (len(pixel) != 2 or not np.isfinite(pixel).all()):
                raise ValueError("invalid predicted pixel")


class AnatomicalTracker:
    """Conservative handedness + past box continuity. Ambiguity withholds; poses are never renamed.

    The 0.5 overlap values and ``max_gap_s`` are inherited, untuned settings; they must be checked on
    labelled frames (overlapping hands, crossings, reappearance). The rules below add no new number.
    """
    def __init__(self, max_gap_s=.25, min_confidence=.3):
        self.last = {}; self.max_gap_s = max_gap_s; self.min_confidence = min_confidence
        self.last_time = -1.

    def assign(self, detections, time_s):
        if time_s <= self.last_time:
            raise ValueError("identity tracker needs increasing acquisition times")
        self.last_time = time_s
        self.last = {s: value for s, value in self.last.items() if time_s-value[0] <= self.max_gap_s}
        # Duplicates are not two hands. Cross-class duplicates are identity-ambiguous; overlap is
        # measured against the smaller box so a box nested inside another counts as a duplicate.
        # Only detections at or above the confidence floor take part: otherwise a box too weak to be a
        # hand could veto a confident hand of the other class it overlapped.
        strong = [k for k, d in enumerate(detections) if d.confidence >= self.min_confidence]
        bad = {i for i in strong for j in strong
               if i != j and box_overlap_of_smaller(detections[i].box, detections[j].box) > .5 and detections[i].side != detections[j].side}
        kept = []
        for i in sorted(range(len(detections)), key=lambda k: -detections[k].confidence):
            d = detections[i]
            if i not in bad and d.confidence >= self.min_confidence and not any(box_iou(d.box, e.box) > .5 for e in kept):
                kept.append(d)
        result = dict.fromkeys(SIDES); conflicts = []
        by_side = {side: [d for d in kept if d.side == side] for side in SIDES}
        for side in SIDES:
            candidates = by_side[side]
            if len(candidates) != 1:
                continue
            d = candidates[0]
            centre = (np.asarray(d.box[:2]) + d.box[2:])/2
            other = SIDES[1-SIDES.index(side)]
            if side in self.last and other in self.last:
                own = np.linalg.norm(centre-self.last[side][1])
                opposite = np.linalg.norm(centre-self.last[other][1])
                if opposite + 10 < own:
                    continue  # potential chirality error/crossing; do not relabel an already mirrored pose
            elif other in self.last and side not in self.last and boxes_touch(d.box, self.last[other][2]):
                # A side appearing exactly where the other hand just was: either that hand with the wrong
                # class, or the earlier track was the wrong one. Accept it only when the other hand is
                # confirmed in this frame at its own place. Otherwise the evidence conflicts: withhold this
                # frame and forget the old track, so the next consistent detection decides (a one-frame loss,
                # at any frame rate, instead of trusting either class).
                confirmed = (len(by_side[other]) == 1 and boxes_touch(by_side[other][0].box, self.last[other][2]))
                if not confirmed:
                    conflicts.append(other)
                    continue
            result[side] = d
        for other in conflicts:
            if result.get(other) is None:
                self.last.pop(other, None)
        for side, d in result.items():
            if d is not None:
                self.last[side] = (time_s, (np.asarray(d.box[:2])+d.box[2:])/2, tuple(d.box))
        return result


DEFAULT_COLOUR_TARGETS = {
    # identity -> list of inclusive HSV (OpenCV scale: H 0-179, S 0-255, V 0-255) ranges; red wraps around hue 0.
    "green_cup": [((35, 70, 35), (85, 255, 255))],
    "red_marker": [((0, 90, 40), (10, 255, 255)), ((170, 90, 40), (179, 255, 255))],
}


class ColourObjects:
    """Declared matte colour targets. Requires a controlled scene and object-specific validation.

    ``targets`` maps an object identity to one or more HSV ranges (default: the original green cup and red
    marker). Ranges are batch settings chosen once on a development clip and recorded with the run
    (``ColourObjects.from_json``); they are never tuned per clip to improve a result."""
    def __init__(self, min_area=100, targets=None):
        self.min_area = min_area
        self.targets = {str(k): [tuple(map(tuple, r)) for r in v] for k, v in (targets or DEFAULT_COLOUR_TARGETS).items()}
        for identity, ranges in self.targets.items():
            if not identity or identity in SIDES or not ranges or any(len(r) != 2 or len(r[0]) != 3 or len(r[1]) != 3 for r in ranges):
                raise ValueError("each colour target needs a non-hand identity and HSV (lower, upper) triples")

    @classmethod
    def from_json(cls, path, min_area=100):
        import json
        data = json.loads(open(path).read())
        return cls(min_area=min_area, targets=data["targets"] if "targets" in data else data)

    def pixels(self, rgb):
        """Legacy: centroid-nearest region pixel without a depth check (kept for old callers)."""
        masks = self._masks(rgb)
        result = {}
        for identity, (region, reason) in self._regions(masks).items():
            if region is None:
                result[identity] = None; continue
            mask, centre = region
            yy, xx = np.nonzero(mask)
            best = np.argmin((xx-centre[0])**2 + (yy-centre[1])**2)
            result[identity] = (int(xx[best]), int(yy[best]))
        return result

    def _masks(self, rgb):
        import cv2
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        # Colour ranges are unvalidated settings: check them on the real objects, hands and lighting.
        masks = {}
        for identity, ranges in self.targets.items():
            mask = None
            for lower, upper in ranges:
                part = cv2.inRange(hsv, tuple(int(v) for v in lower), tuple(int(v) for v in upper))
                mask = part if mask is None else (mask | part)
            masks[identity] = mask
        return masks

    def _regions(self, masks):
        """identity -> ((component mask, centroid) or None, reason); absent and ambiguous are distinct."""
        import cv2
        out = {}
        for identity, mask in masks.items():
            count, labels, stats, centres = cv2.connectedComponentsWithStats(mask)
            candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= self.min_area]
            if not candidates:
                out[identity] = (None, "colour_absent")
            elif len(candidates) > 1:
                out[identity] = (None, "colour_ambiguous_multiple_regions")
            else:
                i = candidates[0]
                out[identity] = ((labels == i, centres[i]), "")
        return out

    def locate(self, rgb, depth, invalid=0):
        """identity -> (pixel or None, reason). The pixel is the one nearest the region's centroid
        that has valid depth, so an object whose centre has no depth (a cup opening) is not lost."""
        result = {}
        for identity, (region, reason) in self._regions(self._masks(rgb)).items():
            if region is None:
                result[identity] = (None, reason); continue
            mask, centre = region
            pixel = _nearest_valid_pixel(np.asarray(depth), mask, centre, invalid)
            result[identity] = (pixel, "" if pixel is not None else "no_valid_depth_in_colour_region")
        return result


def packet_from_detections(frame, assigned, object_pixels, *, participant_id, session_id, available_s, object_details=None):
    """Lift three fixed points, then average in 3-D. All three must have valid local depth.

    Availability includes measured inference completion; acquisition remains that of the camera.
    Objects may be given as pixels (legacy), ``(pixel, reason)`` tuples from ``ColourObjects.locate``, or
    ``ObjectObservation`` records whose metric ``xyz`` is used directly; when ``object_details``
    is a dict it receives each ObjectObservation's details for the diagnostics. An object's name is whatever
    the provider declared; a colour region or mask never silently becomes a different physical object.
    """
    if available_s < frame.available_s:
        raise ValueError("inference cannot be available before its input frame")
    observation = replace(frame.rgb_stamp, sample_id=frame.frame_id+":perception", available_s=available_s)
    hands = []; diagnostics = {}
    for side in SIDES:
        detection = assigned.get(side)
        definition = "missing_three_point_mean" if detection is None else detection.definition+":fixed_3d_mean"
        point = frame.point_at(side, None, observation=observation, definition=definition, decision_s=available_s)
        lifted = []; reasons = {}; point_xyz = {}; extra_xyz = {}
        if detection is not None:
            for name in FUNCTIONAL_POINTS:
                uv = detection.points[name]
                pixel = None if uv is None else tuple(int(round(v)) for v in uv)
                if pixel is not None and not (detection.box[0] <= pixel[0] <= detection.box[2] and detection.box[1] <= pixel[1] <= detection.box[3]):
                    pixel = None
                q = frame.point_at(side+":"+name, pixel, observation=observation,
                                   definition=detection.definition+":"+name, decision_s=available_s)
                reasons[name] = q.reason or "predicted_point_depth_valid_visibility_unverified"
                point_xyz[name] = None if q.xyz is None else [float(v) for v in q.xyz]
                if q.xyz is not None:
                    lifted.append(q.xyz)
            for name, uv in detection.extra_points.items():
                # fingertips: lifted the same way, kept in the diagnostics and the scene record, never in the packet
                pixel = None if uv is None else tuple(int(round(v)) for v in uv)
                if pixel is not None and not (detection.box[0] <= pixel[0] <= detection.box[2] and detection.box[1] <= pixel[1] <= detection.box[3]):
                    pixel = None
                q = frame.point_at(side+":"+name, pixel, observation=observation, definition=detection.definition+":"+name, decision_s=available_s)
                extra_xyz[name] = None if q.xyz is None else [float(v) for v in q.xyz]
        # 0.18 m is an inherited, untuned depth-spread setting: the diagnostics below record the actual
        # point depths and spread so that a limit can be chosen from labelled frames, not guessed.
        spread = float(np.ptp(np.asarray(lifted)[:, 2])) if len(lifted) >= 2 else (0. if lifted else None)
        if len(lifted) == 3 and spread <= .18:
            point = replace(point, xyz=tuple(float(v) for v in np.mean(lifted, axis=0)), reason="")
        else:
            point = replace(point, reason="identity_or_detection_unavailable" if detection is None else "incomplete_or_depth_inconsistent_functional_points")
        hands.append(point); diagnostics[side] = dict(reason=point.reason, points=reasons, point_xyz=point_xyz, extra_xyz=extra_xyz,
                                                      depth_spread_m=spread if detection is not None else None,
                                                      identity_source="model_handedness_and_causal_continuity")
    objects = []
    for identity, value in sorted(object_pixels.items()):
        if isinstance(value, ObjectObservation):
            if value.identity != identity:
                raise ValueError("object observation identity must match its key")
            q = frame.point_from_xyz(identity, value.xyz, observation=observation, definition=value.definition + ":" + identity,
                                     decision_s=available_s, reason=value.reason or "object_unavailable")
            if object_details is not None:
                object_details[identity] = dict(value.details, pixel=value.pixel, reason=q.reason)
            objects.append(q); continue
        # value: a pixel/None (legacy ``pixels``) or (pixel/None, reason) from ``ColourObjects.locate``
        pixel, why = value if isinstance(value, tuple) and len(value) == 2 and (value[0] is None or isinstance(value[0], tuple)) else (value, "")
        q = frame.point_at(identity, pixel, observation=observation,
                           definition="controlled_colour_region_valid_depth_pixel:"+identity, decision_s=available_s)
        objects.append(replace(q, reason=why) if q.xyz is None and why else q)
    objects = tuple(objects)
    return Packet(frame.frame_id, available_s, tuple(hands), objects,
                  clock=frame.rgb_stamp.clock, participant_id=participant_id, session_id=session_id), diagnostics
