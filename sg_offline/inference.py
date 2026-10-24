"""Small proposed joint model and controls. Synthetic fitting is not user validation."""
from dataclasses import dataclass
from itertools import product
from math import fsum
import numpy as np
from .contracts import SIDES, ACTIONS, OUTCOMES
from .state import distance

RELATIONS = ("shared", "symmetric", "unrelated", "unresolved")
FEATURE_DIM = 42
KINETIC_FEATURE_COLUMNS = (3, 4, 18, 19)
FEATURE_MODES = ("motion_only", "motion_and_kinetics")

# ---- feature layouts ------------------------------------------------------------------
# "v1" is the original 42-column layout and stays the default: bit-identical features.
# "v2" appends, before the bias column, one block of ACTION_CUE_NAMES per hand computed on the table-plane anchor and the
# hand landmarks (scene.py): the observations the five non-grasp actions need. Column order of v2:
#   [hand 0: 15 v1 columns][hand 1: 15][shared: 11][hand 0: 14 cues][hand 1: 14 cues][bias]
ACTION_CUE_NAMES = ("hand_height", "hand_height_rate", "height_valid", "vertical_reversals",
                    "object_height", "object_height_rate", "inplane_distance", "sweep_rate",
                    "index_alignment", "grasp_alignment", "landmarks_valid",
                    "over_inplane_distance", "over_height_difference", "over_valid")
# "v3": the v2 block plus the hand aperture from the tracked fingertips, the direct
# kinematic signature of closing and opening. Absent fingertips give aperture_valid 0, never a measured zero.
APERTURE_CUE_NAMES = ("aperture", "aperture_rate", "index_curl", "aperture_valid")
ALL_CUE_NAMES = ACTION_CUE_NAMES + APERTURE_CUE_NAMES
_HAND_ONLY_CUES = (0, 1, 2, 3, 14, 15, 16, 17)   # depend on the hand (and the plane) only
_OBJECT_CUES = tuple(range(4, 14))               # depend on the hypothesis' candidate object


class Layout:
    """Column bookkeeping for one feature layout; every readout sizes itself from it."""
    def __init__(self, name, cues):
        self.name, self.cues_per_hand = name, cues
        self.v1_per_hand, self.v1_shared = 15, 11       # shared v1 columns without the bias
        self.dim = 2 * self.v1_per_hand + self.v1_shared + 2 * cues + 1
        self.bias = self.dim - 1

    def own(self, hand):
        """Every column that belongs to one hand: its v1 block and, under v2, its cue block (no bias)."""
        return list(range(hand * 15, (hand + 1) * 15)) + self.cues(hand)

    def cues(self, hand):
        start = 2 * self.v1_per_hand + self.v1_shared + hand * self.cues_per_hand
        return list(range(start, start + self.cues_per_hand))

    @property
    def shared(self):
        """The v1 shared block plus the bias (as the original code read it)."""
        return list(range(30, 42)) if self.cues_per_hand == 0 else list(range(30, 41)) + [self.bias]

    def hand_only(self, hand):
        cues = self.cues(hand)
        return [hand * 15 + c for c in _HAND_COLUMNS] + [cues[c] for c in _HAND_ONLY_CUES if c < len(cues)]

    def object_dependent(self, hand):
        cues = self.cues(hand)
        return [hand * 15 + c for c in _OBJECT_COLUMNS] + [cues[c] for c in _OBJECT_CUES if c < len(cues)]


LAYOUTS = {"v1": Layout("v1", 0), "v2": Layout("v2", len(ACTION_CUE_NAMES)), "v3": Layout("v3", len(ALL_CUE_NAMES))}


def layout_of(name):
    if name not in LAYOUTS:
        raise ValueError(f"unknown feature layout {name!r}; known: {sorted(LAYOUTS)}")
    return LAYOUTS[name]


def softmax(z):
    z = np.asarray(z, dtype=float)
    if not np.isfinite(z).all():
        raise ValueError("nonfinite model score")
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def log_softmax(z):
    """Keep finite, confidently wrong logits trainable even when their probabilities underflow."""
    z = np.asarray(z, dtype=float)
    if not np.isfinite(z).all():
        raise ValueError("nonfinite model score")
    shifted = z - z.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def allowed_log_mass(logits, label):
    return np.logaddexp.reduce(log_softmax(logits)[..., label_indices(label)], axis=-1)


def logits_residual(logits, label):
    allowed = label_indices(label)
    residual = softmax(logits)
    residual[..., allowed] -= softmax(np.asarray(logits)[..., allowed])
    return residual


@dataclass(frozen=True)
class Hypothesis:
    object_ids: tuple[str | None, str | None]
    relation: str
    roles: tuple[str, str]
    weight: float
    candidate_truncated: bool = False


@dataclass(frozen=True)
class Bundle:
    hypotheses: tuple[Hypothesis, ...]
    features: np.ndarray  # H x 42
    weights: np.ndarray  # H
    source_kinds: tuple[str, ...]
    feature_mode: str = "unspecified"
    layout: str = "v1"

    def __post_init__(self):
        if self.feature_mode not in (*FEATURE_MODES, "unspecified"):
            raise ValueError("unknown feature mode")
        if self.features.shape != (len(self.hypotheses), layout_of(self.layout).dim) or self.weights.shape != (len(self.hypotheses),):
            raise ValueError("hypothesis/feature/weight alignment")
        if not len(self.weights) or not np.isfinite(self.features).all() or not np.isfinite(self.weights).all():
            raise ValueError("empty or nonfinite bundle")
        if (self.weights < 0).any() or not np.isclose(self.weights.sum(), 1):
            raise ValueError("weights must form a normalized nonnegative mixture")
        if not np.allclose(self.weights, [h.weight for h in self.hypotheses]):
            raise ValueError("hypothesis provenance weight mismatch")
        if not self.source_kinds or any(s not in ("synthetic", "recorded", "measured") for s in self.source_kinds):
            raise ValueError("explicit upstream source kinds required")
        if self.feature_mode == "motion_only" and np.any(self.features[:, KINETIC_FEATURE_COLUMNS] != 0):
            raise ValueError("motion-only bundles cannot contain kinetic values or validity flags")
        for key in ("features", "weights"):
            value = np.array(getattr(self, key), dtype=float, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, key, value)


def hypotheses(packet, motions, loads, kinetics=True, max_age_s=.25, role_mode="weak_speed", relatives=None, cues=None, layout="v1"):
    """Transparent uncalibrated geometry/role proposal; keeps null and competing relations.

    ``cues`` (``scene.SceneCues``) and ``layout="v2"`` add the per-action cue block; with the default
    layout the bundle is the 42-column bundle of every earlier record, cues or not."""
    layout_of(layout)
    if role_mode not in ("weak_speed", "neutral", "permuted"):
        raise ValueError("unknown role-prior mode")
    objects = {p.identity: p for p in packet.objects if p.xyz is not None
               and packet.decision_s - p.stamp.acquired_s <= max_age_s}
    choices = []
    for hand in packet.hands:
        if hand.xyz is None or packet.decision_s - hand.stamp.acquired_s > max_age_s:
            choices.append([(None, 1.0)])
            continue
        compatible = [o for o in objects.values() if o.stamp.acquired_s == hand.stamp.acquired_s
                      and o.frame == hand.frame and o.calibration_id == hand.calibration_id]
        ranked = sorted(compatible, key=lambda o: (distance(hand.xyz, o.xyz), o.identity))
        candidates = [(o.identity, np.exp(-distance(hand.xyz, o.xyz)/0.1)) for o in ranked[:2]]
        omitted_score = sum(np.exp(-distance(hand.xyz, o.xyz)/0.1) for o in ranked[2:])
        candidates.append((None, 0.25 + omitted_score))  # retained unknown/outside score, not probability coverage
        total = sum(w for _, w in candidates)
        choices.append([(key, float(w/total)) for key, w in candidates])
    entries = []
    for (left, wl), (right, wr) in product(*choices):
        if left is None or right is None:
            relations = [("unresolved", ("other", "other"), 1.)]
        elif left == right:
            # Weak speed-based prior allows moving support and changing roles.
            # It is an uncalibrated comparator assumption, not a reference label.
            velocities = [motions[s].velocity_mps for s in SIDES]
            support_left = .5
            if role_mode != "neutral" and all(v is not None for v in velocities):
                speeds = [np.linalg.norm(v) for v in velocities]
                support_left = .1 + .8 / (1 + np.exp(np.clip((speeds[0]-speeds[1])/.2, -30, 30)))
            if role_mode == "permuted":
                support_left = 1-support_left
            relations = [("shared", ("support", "manipulate"), .7*support_left),
                         ("shared", ("manipulate", "support"), .7*(1-support_left)),
                         ("symmetric", ("manipulate", "manipulate"), .2),
                         ("unrelated", ("other", "other"), .1)]
        else:
            relations = [("unrelated", ("other", "other"), .7),
                         ("shared", ("support", "manipulate"), .15),
                         ("shared", ("manipulate", "support"), .15)]
        for relation, roles, w in relations:
            entries.append(Hypothesis((left, right), relation, roles, wl*wr*w, len(objects) > 2))
    weights = np.array([h.weight for h in entries]); weights /= weights.sum()
    entries = tuple(Hypothesis(h.object_ids, h.relation, h.roles, float(w), h.candidate_truncated) for h, w in zip(entries, weights))
    features = np.vstack([encode(packet, motions, loads, h, kinetics, relatives, cues, layout) for h in entries])
    active_records = packet.hands + packet.objects
    if kinetics:
        active_records += packet.telemetry
    source_kinds = tuple(sorted({record.stamp.source.value for record in active_records}))
    feature_mode = "motion_and_kinetics" if kinetics else "motion_only"
    return Bundle(entries, features, weights, source_kinds, feature_mode, layout)


def label_indices(label):
    """Joint outcomes consistent with known hand references; unknown hands are marginalised."""
    values = (int(label),) if isinstance(label, (int, np.integer)) else tuple(label)
    if not values or len(set(values)) != len(values) or any(type(v) is not int or not 0 <= v < 49 for v in values):
        raise ValueError("nonempty distinct allowed joint-label indices required")
    return list(values)


def probability_residual(p, label):
    allowed = label_indices(label)
    mass = np.maximum(p[..., allowed].sum(axis=-1, keepdims=True), 1e-300)
    residual = p.copy()
    residual[..., allowed] -= p[..., allowed] / mass
    return residual


# ---- full-information flat comparator input ------------------------------------------------------
# Per hand, the 15 encoded columns split into object-dependent ones and hand-only ones (see ``encode``).
_OBJECT_COLUMNS = (0, 1, 6, 8)                      # distance, approach rate, object valid, rate valid
_HAND_COLUMNS = (2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14)  # speed, load, load valid, hand valid, age, past commands
_SAME = (("shared", ("support", "manipulate")), ("shared", ("manipulate", "support")),
         ("symmetric", ("manipulate", "manipulate")), ("unrelated", ("other", "other")))
_CROSS = (("unrelated", ("other", "other")), ("shared", ("support", "manipulate")), ("shared", ("manipulate", "support")))
# One row per object: left features+weight, right features+weight, then the weight of every hypothesis
# that involves this object (both hands on it by relation; left on it and right on the other object by
# relation; left on it and right on nothing; right on it and left on nothing).
_ROW_DIM = 2 * (len(_OBJECT_COLUMNS) + 1) + len(_SAME) + len(_CROSS) + 2
_TRIU = np.triu_indices(_ROW_DIM)
FLAT_DIM = 2 * len(_HAND_COLUMNS) + 3 + _ROW_DIM + len(_TRIU[0])


def _flat_spec(layout):
    """(object columns per hand, hand-only columns per hand, row dim, triu indices, flat dim) for a layout."""
    lay = layout_of(layout)
    per_hand_object = len(lay.object_dependent(0))
    row_dim = 2 * (per_hand_object + 1) + len(_SAME) + len(_CROSS) + 2
    triu = np.triu_indices(row_dim)
    flat = 2 * len(lay.hand_only(0)) + 3 + row_dim + len(triu[0])
    return per_hand_object, row_dim, triu, flat


def flat_dim(layout="v1"):
    return _flat_spec(layout)[3]


def flat_object_rows(bundle):
    """The two per-object rows the flat input encodes (a zero row stands for a missing object)."""
    H, F, w = bundle.hypotheses, bundle.features, bundle.weights
    lay = layout_of(bundle.layout)
    n_obj, row_dim, _, _ = _flat_spec(bundle.layout)
    objects = sorted({o for h in H for o in h.object_ids if o is not None})
    if len(objects) > 2:
        raise ValueError("the flat comparator covers the controlled scene: at most two objects among the hands' candidates")
    rows = []
    for o in objects:
        # Weights are summed with math.fsum (exactly rounded), so hypothesis order cannot change a bit.
        r = np.zeros(row_dim); parts = {}
        for hand, base in ((0, 0), (1, n_obj + 1)):
            for h, weight, row in zip(H, w, F):
                if h.object_ids[hand] == o:
                    r[base:base+n_obj] = row[np.asarray(lay.object_dependent(hand))]
                    parts.setdefault(base+n_obj, []).append(float(weight))
        offset = 2 * (n_obj + 1)
        for h, weight in zip(H, w):
            left, right = h.object_ids; option = (h.relation, h.roles)
            if left == o and right == o:
                index = offset + _SAME.index(option)
            elif left == o and right is not None:
                index = offset + len(_SAME) + _CROSS.index(option)
            elif left == o and right is None:
                index = offset + len(_SAME) + len(_CROSS)
            elif left is None and right == o:
                index = offset + len(_SAME) + len(_CROSS) + 1
            else:
                continue
            parts.setdefault(index, []).append(float(weight))
        for index, values in parts.items():
            r[index] = fsum(values)
        rows.append(r)
    while len(rows) < 2:
        rows.append(np.zeros(row_dim))
    return rows


def flat_features(bundle):
    """Full-information flat input that is continuous in the scene and blind to object names.

    The scene's (at most two) objects are an unordered pair of rows r1, r2 (see ``flat_object_rows``).
    Fixed slots for an unordered pair either jump when the objects swap rank or, if averaged, lose which
    value belongs to which object at a tie. This input instead keeps the sum r1 + r2 and every product of
    the difference (r1 - r2)(r1 - r2)^T: both are continuous, unchanged by swapping or renaming the
    objects, and together they determine r1 and r2 exactly (``flat_rows_from_features``). Hand-only
    features, the both-hands-on-nothing weight, inter-hand distance and both-hands-valid complete it.
    """
    H, F, w = bundle.hypotheses, bundle.features, bundle.weights
    lay = layout_of(bundle.layout)
    _, _, triu, _ = _flat_spec(bundle.layout)
    r1, r2 = flat_object_rows(bundle)
    d = r1 - r2
    none_none = fsum(float(weight) for h, weight in zip(H, w) if h.object_ids == (None, None))
    return np.concatenate([F[0, np.asarray(lay.hand_only(0))], F[0, np.asarray(lay.hand_only(1))], [none_none],
                           F[0, [30, 31]], r1 + r2, np.outer(d, d)[triu]])


def flat_rows_from_features(vector, layout="v1"):
    """Recover the unordered pair of object rows from a flat input (the completeness check)."""
    lay = layout_of(layout)
    _, row_dim, triu, _ = _flat_spec(layout)
    start = 2 * len(lay.hand_only(0)) + 3
    total = np.asarray(vector[start:start + row_dim])
    D = np.zeros((row_dim, row_dim)); D[triu] = vector[start + row_dim:]
    D = D + np.triu(D, 1).T
    k = int(np.argmax(np.diag(D)))
    d = D[:, k] / np.sqrt(D[k, k]) if D[k, k] > 0 else np.zeros(row_dim)
    return (total + d) / 2, (total - d) / 2


def hand_target_features(bundle, hand):
    """Most-supported non-null object, independent of relation multiplicity and null weight."""
    groups = {}
    for j, h in enumerate(bundle.hypotheses):
        if h.object_ids[hand] is not None:
            groups.setdefault(h.object_ids[hand], []).append(j)
    if not groups:
        return None
    # Sums are rounded before comparison: at an exact distance tie two objects' sums can differ in the last bit, so the
    # name tie-break would never be reached and the target would follow the order of summation.
    identity = min(groups, key=lambda k: (-round(float(bundle.weights[groups[k]].sum()), 12), k))
    return bundle.features[groups[identity][0], hand*15:(hand+1)*15]


def hand_action_cues(bundle, hand):
    """The action-cue block (ACTION_CUE_NAMES) of the most-supported object's row for one hand; None under layout v1
    or when the hand has no object candidate."""
    lay = layout_of(bundle.layout)
    if lay.cues_per_hand == 0:
        return None
    groups = {}
    for j, h in enumerate(bundle.hypotheses):
        if h.object_ids[hand] is not None:
            groups.setdefault(h.object_ids[hand], []).append(j)
    if not groups:
        return None
    identity = min(groups, key=lambda k: (-round(float(bundle.weights[groups[k]].sum()), 12), k))
    return bundle.features[groups[identity][0], np.asarray(lay.cues(hand))]


def cue(block, name):
    """A named cue of a block (v2 blocks are a prefix of v3 blocks, so names index both)."""
    index = ALL_CUE_NAMES.index(name)
    if index >= len(block):
        raise ValueError(f"cue {name!r} is not in this layout's block")
    return float(block[index])


def fit_checkpoint(model, bundles, labels, step):
    loss = float(np.mean([-model.log_likelihood(b, y)
                          for b,y in zip(bundles,labels)]))
    if not np.isfinite(loss):
        raise ValueError("nonfinite fitting loss")
    if step == 0:
        model.fit_diagnostics = dict(optimizer="Adam", loss_curve=[])
    model.fit_diagnostics["loss_curve"].append(dict(step=step, unregularized_log_loss=loss))


class Adam:
    """Small deterministic full-batch optimiser shared by the numpy readouts."""
    def __init__(self, params):
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, gradients, lr):
        self.t += 1
        for p, g, m, v in zip(params, gradients, self.m, self.v):
            m *= .9; m += .1*g
            v *= .999; v += .001*g*g
            p -= lr*(m/(1-.9**self.t))/(np.sqrt(v/(1-.999**self.t))+1e-8)


def validate_synthetic_training(bundles, labels):
    if not bundles or len(bundles) != len(labels):
        raise ValueError("aligned nonempty synthetic training set required")
    if any(b.source_kinds != ("synthetic",) for b in bundles):
        raise ValueError("synthetic trainer rejects recorded/measured source bundles")
    if any(not isinstance(y, (int, np.integer)) or not 0 <= y < 49 for y in labels):
        raise ValueError("known integer paired labels required")
    modes = {bundle.feature_mode for bundle in bundles}
    if len(modes) != 1 or not modes.issubset(FEATURE_MODES):
        raise ValueError("training requires one explicit feature mode")


def validate_model_mode(model, bundle):
    if model.training_feature_mode is not None and bundle.feature_mode != model.training_feature_mode:
        raise ValueError("model training feature mode differs from inference inputs; train a separate model")
    layout = getattr(model, "layout", None)
    if layout is not None and layout != bundle.layout:
        raise ValueError(f"model feature layout {layout!r} differs from the bundle layout {bundle.layout!r}; build the model for this layout")


def prepare_synthetic_training(model, bundles, labels):
    validate_synthetic_training(bundles, labels)
    for bundle in bundles:
        validate_model_mode(model, bundle)


def encode(packet, motions, loads, h, kinetics, relatives=None, cues=None, layout="v1"):
    lay = layout_of(layout)
    rows = []
    for i, side in enumerate(SIDES):
        motion = motions[side]; p = motion.point
        obj = motions.get(h.object_ids[i])
        valid = p.xyz is not None
        ov = obj is not None and obj.point.xyz is not None
        vv = valid and ov and motion.velocity_mps is not None and obj.velocity_mps is not None
        d = distance(p.xyz, obj.point.xyz) if valid and ov else 0.
        radial = 0.
        # Same physical quantity, d|hand-object|/dt, from same-time distances: needs no fixed frame.
        rel = relatives.get((side, h.object_ids[i])) if relatives and h.object_ids[i] is not None else None
        if rel is not None and rel.rate_mps is not None and rel.reason.startswith("past_least_squares"):
            # A declared rate window (Config.rate_window > 1) governs the feature in every frame, also
            # for fixed-frame points, which would otherwise keep the one-frame projection below. With the
            # default window 1 the relative rate is a secant and this branch is never taken.
            radial, vv = rel.rate_mps, True
        elif vv and d > 1e-9:
            delta = np.asarray(p.xyz) - obj.point.xyz
            radial = float(np.dot(delta/d, np.asarray(motion.velocity_mps)-obj.velocity_mps))
        elif rel is not None and rel.rate_mps is not None:
            radial, vv = rel.rate_mps, True
        speed = float(np.linalg.norm(motion.velocity_mps)) if motion.velocity_mps is not None else 0.
        load = loads[i] if kinetics else None
        lv = load is not None and load.value_N is not None
        rows.extend([d/.1, radial/.2, speed/.2, load.value_N/10 if lv else 0., float(lv),
                     float(valid), float(ov), min(motion.age_s/.25, 4.), float(vv)])
        rows.extend(float(packet.previous_commands[i] == a) for a in ACTIONS)
    both = (all(p.xyz is not None for p in packet.hands)
            and packet.hands[0].stamp.acquired_s == packet.hands[1].stamp.acquired_s
            and packet.hands[0].calibration_id == packet.hands[1].calibration_id)
    rows.extend([distance(packet.hands[0].xyz, packet.hands[1].xyz)/.2 if both else 0., float(both)])
    rows.append(float(h.object_ids[0] is not None and h.object_ids[0] == h.object_ids[1]))
    rows.extend(float(h.relation == r) for r in RELATIONS)
    rows.extend(float(role == r) for role in h.roles for r in ("support", "manipulate"))
    if lay.cues_per_hand:
        for i, side in enumerate(SIDES):
            rows.extend(action_cues(packet, cues, side, h.object_ids[i], lay.cues_per_hand))
    rows.append(1.)
    return np.array(rows)


def _clip(v, lo, hi):
    return float(min(max(v, lo), hi))


def _aperture_cues(cues, side):
    block = [0.] * len(APERTURE_CUE_NAMES)
    track = None if cues is None else cues.apertures.get(side)
    if track is None or track.aperture_m is None:
        return block
    block[0] = _clip(track.aperture_m / .1, 0., 5.)
    block[1] = _clip((track.rate_mps or 0.) / .2, -10., 10.)
    block[2] = _clip(track.index_curl if track.index_curl is not None else 0., 0., 5.)
    block[3] = 1.
    return block


def action_cues(packet, cues, side, object_id, size=len(ACTION_CUE_NAMES)):
    """The cue block for one hand and its candidate object (zeros with validity flags down when absent); ``size`` 14 (v2)
    or 18 (v3, with the aperture block, which needs no plane)."""
    if size not in (len(ACTION_CUE_NAMES), len(ALL_CUE_NAMES)):
        raise ValueError("unknown cue block size")
    tail = _aperture_cues(cues, side) if size == len(ALL_CUE_NAMES) else []
    block = [0.] * len(ACTION_CUE_NAMES)
    if cues is None or not cues.plane_available:
        return block + tail
    hand = cues.heights.get(side)
    if hand is not None and hand.height_m is not None:
        block[0] = _clip(hand.height_m / .2, -2.5, 10.)
        block[1] = _clip((hand.rate_mps or 0.) / .2, -10., 10.)
        block[2] = 1.
        reversals = cues.reversals.get(side)
        block[3] = _clip((reversals or 0) / 4., 0., 4.)
    if object_id is None:
        return block + tail
    obj = cues.heights.get(object_id)
    pair = cues.pairs.get((side, object_id))
    if obj is not None and obj.height_m is not None and pair is not None and pair.inplane_distance_m is not None:
        block[4] = _clip(obj.height_m / .2, -2.5, 10.)
        block[5] = _clip((obj.rate_mps or 0.) / .2, -10., 10.)
        block[6] = _clip(pair.inplane_distance_m / .1, 0., 20.)
        block[7] = _clip((pair.sweep_rate_rad_s or 0.) / 2., -10., 10.)
    hand_point = next(p for p in packet.hands if p.identity == side)
    object_point = next((p for p in packet.objects if p.identity == object_id), None)
    from .scene import landmark_alignments  # local import: scene imports contracts only, inference stays importable alone
    index_align, grasp_align, valid = landmark_alignments(cues.landmarks.get(side), hand_point.xyz,
                                                          None if object_point is None else object_point.xyz)
    block[8], block[9], block[10] = index_align, grasp_align, float(valid)
    over = cues.over.get(object_id)
    if over is not None and over.inplane_distance_m is not None:
        block[11] = _clip(over.inplane_distance_m / .1, 0., 20.)
        block[12] = _clip(over.height_difference_m / .2, -10., 10.)
        block[13] = 1.
    return block + tail


class LinearJoint:
    """KT-R (``mixture=True``): one 49x42 softmax readout applied to EVERY retained hypothesis, then
    pooled by the hypothesis weights. ``mixture=False`` is its mean-collapse twin: the same readout
    applied once to the weighted-mean feature vector (the hypothesis set collapsed before the readout).
    Together with ``HardCollapseLinear`` this isolates whether keeping the competing hypotheses matters."""
    def __init__(self, mixture=True, layout="v1"):
        self.layout = layout_of(layout).name
        self.W = np.zeros((49, layout_of(layout).dim))
        self.mixture = mixture
        self.name = "relational_mixture" if mixture else "direct_linear"
        self.fit_provenance = "untrained"
        self.training_feature_mode = None

    def _inputs(self, bundle):
        validate_model_mode(self, bundle)
        if self.mixture:
            return bundle.features, bundle.weights
        return (bundle.weights @ bundle.features)[None, :], np.ones(1)

    def predict(self, bundle):
        x, weights = self._inputs(bundle)
        return weights @ softmax(x @ self.W.T)

    def log_likelihood(self, bundle, label):
        x, weights = self._inputs(bundle)
        log_weights = np.log(weights, out=np.full_like(weights, -np.inf), where=weights > 0)
        return float(np.logaddexp.reduce(log_weights + allowed_log_mass(x @ self.W.T, label)))

    def loss_gradient(self, bundle, label, regularization=0.):
        allowed = label_indices(label)
        x, weights = self._inputs(bundle)
        logits = x @ self.W.T
        log_weights = np.log(weights, out=np.full_like(weights, -np.inf), where=weights > 0)
        joint_log_mass = log_weights + allowed_log_mass(logits, label)
        observed_log = float(np.logaddexp.reduce(joint_log_mass))
        responsibility = np.exp(joint_log_mass - observed_log)
        residual = logits_residual(logits, label)
        grad = (responsibility[:, None] * residual).T @ x + regularization*self.W
        loss = -observed_log + regularization*np.sum(self.W*self.W)/2
        return float(loss), grad

    def _fit(self, bundles, labels, steps, lr, regularization, callback=None):
        if steps < 1 or lr <= 0 or regularization < 0:
            raise ValueError("invalid fit settings")
        opt = Adam([self.W])
        fit_checkpoint(self, bundles, labels, 0)
        self.training_feature_mode = bundles[0].feature_mode
        for step in range(steps):
            grads = [self.loss_gradient(b, y, regularization)[1] for b, y in zip(bundles, labels)]
            opt.step([self.W], [np.mean(grads, axis=0)], lr)
            if (step+1) % max(1,steps//10) == 0 or step+1 == steps:
                fit_checkpoint(self, bundles, labels, step+1)
            if callback is not None and callback(step+1):
                break  # the caller's early-stopping rule (development patience) asked to stop

    def fit_synthetic(self, bundles, labels, steps=100, lr=.02, regularization=.001):
        prepare_synthetic_training(self, bundles, labels)
        self._fit(bundles, labels, steps, lr, regularization)
        self.fit_provenance = "synthetic_only_no_reference_validation"


class MLPJoint:
    """2051-parameter nonlinear direct control versus 2058-parameter mixture readout."""
    name = "direct_mlp22"

    def __init__(self, seed=0, layout="v1"):
        self.layout = layout_of(layout).name
        self.in_dim = layout_of(layout).dim - 1  # the constant bias column is dropped (the MLPs carry their own biases)
        rng = np.random.default_rng(seed)
        self.A = rng.normal(0, .03, (self.in_dim, 22)); self.a = np.zeros(22)
        self.B = rng.normal(0, .03, (22, 49)); self.b = np.zeros(49)
        self.fit_provenance = "untrained"
        self.training_feature_mode = None

    def _vector(self, bundle):
        return (bundle.weights @ bundle.features)[:self.in_dim]

    def _forward(self, bundle):
        validate_model_mode(self, bundle)
        x = self._vector(bundle)
        h = np.tanh(x @ self.A + self.a)
        return x, h, softmax(h @ self.B + self.b)

    def predict(self, bundle):
        return self._forward(bundle)[2]

    def log_likelihood(self, bundle, label):
        _, h, _ = self._forward(bundle)
        return float(allowed_log_mass(h @ self.B + self.b, label))

    def _fit(self, bundles, labels, steps, lr, regularization, callback=None):
        if steps < 1 or lr <= 0 or regularization < 0:
            raise ValueError("invalid fit settings")
        params = (self.A, self.a, self.B, self.b)
        opt = Adam(params)
        fit_checkpoint(self, bundles, labels, 0)
        self.training_feature_mode = bundles[0].feature_mode
        for step in range(steps):
            gradients = []
            for bundle, y in zip(bundles, labels):
                x, h, p = self._forward(bundle)
                dy = logits_residual(h @ self.B + self.b, y)
                dh = (self.B @ dy)*(1-h*h)
                gradients.append((np.outer(x, dh), dh, np.outer(h, dy), dy))
            opt.step(params, [np.mean(g, axis=0) + regularization*param
                              for param, g in zip(params, zip(*gradients))], lr)
            if (step+1) % max(1,steps//10) == 0 or step+1 == steps:
                fit_checkpoint(self, bundles, labels, step+1)
            if callback is not None and callback(step+1):
                break  # the caller's early-stopping rule (development patience) asked to stop

    def fit_synthetic(self, bundles, labels, steps=100, lr=.02, regularization=.001):
        prepare_synthetic_training(self, bundles, labels)
        self._fit(bundles, labels, steps, lr, regularization)
        self.fit_provenance = "synthetic_only_no_reference_validation"


class FlatJoint(MLPJoint):
    """Information-matched flat comparator (see ``flat_features``).

    The hidden width (7) gives 2,037 parameters against KT-SET's 2,051, and weights start with the
    same N(0, 0.03) rule as KT-SET, so the two differ in representation, not in size or initialisation.
    """
    name = "full_information_flat_mlp"

    def __init__(self, seed=0, hidden=7, layout="v1"):
        super().__init__(seed, layout)
        rng = np.random.default_rng(seed)
        self.A = rng.normal(0, .03, (flat_dim(layout), hidden))
        self.a = np.zeros(hidden)
        self.B = rng.normal(0, .03, (hidden, 49)); self.b = np.zeros(49)

    def _vector(self, bundle):
        return flat_features(bundle)


class HypothesisPoolMLP(MLPJoint):
    """Same entire hypothesis-input set, nonlinear embedding BEFORE weighted pooling."""
    name = "hypothesis_pool_mlp22"

    def _forward_set(self, bundle):
        validate_model_mode(self, bundle)
        x = bundle.features[:, :self.in_dim]
        hidden = np.tanh(x @ self.A + self.a)
        pooled = bundle.weights @ hidden
        return x, hidden, pooled, softmax(pooled @ self.B + self.b)

    def predict(self, bundle):
        return self._forward_set(bundle)[3]

    def log_likelihood(self, bundle, label):
        _, _, pooled, _ = self._forward_set(bundle)
        return float(allowed_log_mass(pooled @ self.B + self.b, label))

    def _fit(self, bundles, labels, steps, lr, regularization, callback=None):
        if steps < 1 or lr <= 0 or regularization < 0:
            raise ValueError("invalid fit settings")
        params = (self.A, self.a, self.B, self.b)
        opt = Adam(params)
        fit_checkpoint(self, bundles, labels, 0)
        self.training_feature_mode = bundles[0].feature_mode
        for step in range(steps):
            gradients = []
            for bundle, y in zip(bundles, labels):
                x, h, pooled, p = self._forward_set(bundle)
                dy = logits_residual(pooled @ self.B + self.b, y)
                dh = bundle.weights[:, None] * (self.B @ dy)[None, :] * (1-h*h)
                gradients.append((x.T @ dh, dh.sum(axis=0), np.outer(pooled, dy), dy))
            opt.step(params, [np.mean(g, axis=0) + regularization*param
                              for param, g in zip(params, zip(*gradients))], lr)
            if (step+1) % max(1,steps//10) == 0 or step+1 == steps:
                fit_checkpoint(self, bundles, labels, step+1)
            if callback is not None and callback(step+1):
                break  # the caller's early-stopping rule (development patience) asked to stop

    def fit_synthetic(self, bundles, labels, steps=100, lr=.02, regularization=.001):
        prepare_synthetic_training(self, bundles, labels)
        self._fit(bundles, labels, steps, lr, regularization)
        self.fit_provenance = "synthetic_only_no_reference_validation"


HARD_TIE_TOLERANCE = 1e-9  # weights equal within this are a tie for the hard-collapse twin (plan default)


class HardCollapseLinear(LinearJoint):
    """Hard-collapse twin of KT-R: the same 49x42 readout applied to the single most-supported hypothesis
    (only exact ties, weights equal within 1e-9, are averaged, so the input does not depend on hypothesis
    order; the plan default replaced an earlier ``isclose`` reading, which merged
    near-ties such as 0.500001 / 0.499999). Only the top association
    survives; with the mean-collapse twin (``LinearJoint(False)``) it isolates the retained-hypotheses
    mechanism (same 2,058 parameters as KT-R)."""
    def __init__(self, layout="v1"):
        super().__init__(mixture=False, layout=layout)
        self.name = "hard_collapse_linear"

    def _inputs(self, bundle):
        validate_model_mode(self, bundle)
        top = np.abs(bundle.weights - bundle.weights.max()) <= HARD_TIE_TOLERANCE
        return bundle.features[top].mean(axis=0)[None, :], np.ones(1)


# ---- learned simple controls: independent hands (M-I) and explicit coordination (M-C) -------------
# ``encode`` writes 15 columns per hand at hand*15 (distance, approach rate, speed, load, load valid,
# hand valid, object valid, age, rate valid, six past-command flags); columns 30-41 are shared
# (inter-hand distance, both valid, same-object flag, relation and role one-hots, bias).
_OWN_COLUMNS = (tuple(range(0, 15)), tuple(range(15, 30)))
_SHARED_COLUMNS = tuple(range(30, 42))


class FactorizedJoint:
    """Learned per-hand readouts whose joint log-score is a sum of two hand scores, plus for M-C an
    explicit 7x7 pair-coupling matrix:  log p(l, r) = z_left(l) + z_right(r) [+ C(l, r)] - log Z.

    * M-I (``cross_hand=False, coupling=False``): each hand sees only its own 15 columns and a bias, so
      the joint is the product of two independent per-hand softmaxes (224 parameters).
    * M-C (``cross_hand=True, coupling=True``): each hand also sees the other hand's columns and the
      shared coordination columns, and C adds a learned pairwise coordination term (637 parameters).

    Both are fitted on the same labels, passes, learning rate, regularisation and development rule as the
    relational readouts. They replace the hand-authored ``SimpleRules`` as the simple controls:
    the rules make identical decisions at every grid threshold and never fit anything, so
    "KT-R beats the rules" could not be read as "a learned model beats a simpler learned model".
    """

    def __init__(self, cross_hand=False, coupling=False, layout="v1"):
        self.cross_hand, self.coupling = bool(cross_hand), bool(coupling)
        self.layout = layout_of(layout).name
        lay = layout_of(layout)
        own = len(lay.own(0))
        width = 2 * own + len(lay.shared) if self.cross_hand else own + 1   # v1: 42 or 16, as before
        self.W_left = np.zeros((7, width)); self.W_right = np.zeros((7, width))
        self.C = np.zeros((7, 7)) if self.coupling else None
        self.name = "learned_coordinated_hands" if (self.cross_hand or self.coupling) else "learned_independent_hands"
        self.fit_provenance = "untrained"
        self.training_feature_mode = None

    def _params(self):
        return [self.W_left, self.W_right] + ([self.C] if self.coupling else [])

    def n_parameters(self):
        return int(sum(p.size for p in self._params()))

    def get_state(self):
        return [p.copy() for p in self._params()]

    def set_state(self, state):
        for p, value in zip(self._params(), state):
            p[...] = value

    def _inputs(self, bundle):
        validate_model_mode(self, bundle)
        lay = layout_of(bundle.layout)
        x = bundle.weights @ bundle.features
        own = [x[lay.own(0)], x[lay.own(1)]]
        if not self.cross_hand:
            return [np.concatenate([own[i], [1.]]) for i in range(2)]
        shared = x[lay.shared]
        return [np.concatenate([own[i], own[1 - i], shared]) for i in range(2)]

    def _logits(self, bundle):
        u = self._inputs(bundle)
        z_left, z_right = self.W_left @ u[0], self.W_right @ u[1]
        logits = z_left[:, None] + z_right[None, :]
        if self.coupling:
            logits = logits + self.C
        return u, logits

    def predict(self, bundle):
        return softmax(self._logits(bundle)[1].ravel())

    def log_likelihood(self, bundle, label):
        return float(allowed_log_mass(self._logits(bundle)[1].ravel(), label))

    def loss_gradient(self, bundle, label, regularization=0.):
        allowed = label_indices(label)
        u, logits = self._logits(bundle)
        log_mass = allowed_log_mass(logits.ravel(), label)
        d_logits = logits_residual(logits.ravel(), label).reshape(7, 7)
        grads = [np.outer(d_logits.sum(axis=1), u[0]) + regularization*self.W_left,
                 np.outer(d_logits.sum(axis=0), u[1]) + regularization*self.W_right]
        if self.coupling:
            grads.append(d_logits + regularization*self.C)
        loss = -log_mass + regularization*sum(float(np.sum(q*q)) for q in self._params())/2
        return float(loss), grads

    def _fit(self, bundles, labels, steps, lr, regularization, callback=None):
        if steps < 1 or lr <= 0 or regularization < 0:
            raise ValueError("invalid fit settings")
        params = self._params()
        opt = Adam(params)
        fit_checkpoint(self, bundles, labels, 0)
        self.training_feature_mode = bundles[0].feature_mode
        for step in range(steps):
            grads = [self.loss_gradient(b, y, regularization)[1] for b, y in zip(bundles, labels)]
            opt.step(params, [np.mean(g, axis=0) for g in zip(*grads)], lr)
            if (step+1) % max(1, steps//10) == 0 or step+1 == steps:
                fit_checkpoint(self, bundles, labels, step+1)
            if callback is not None and callback(step+1):
                break  # the caller's early-stopping rule (development patience) asked to stop

    def fit_synthetic(self, bundles, labels, steps=100, lr=.02, regularization=.001):
        prepare_synthetic_training(self, bundles, labels)
        self._fit(bundles, labels, steps, lr, regularization)
        self.fit_provenance = "synthetic_only_no_reference_validation"


class SimpleRules:
    """Deliberately small, unvalidated causal rule controls, not external method reproduction.

    These are a DESCRIPTIVE FLOOR, not the simple controls: the per-hand grasp
    score takes only the values 0.0106 and 0.8617 (0.8991 with coordination), so M-I and M-C decide
    identically at every threshold of the 0.3-0.9 grid and never fit anything. ``FactorizedJoint`` is
    the fitted M-I / M-C pair."""
    def __init__(self, coordination=False, kinetics=False):
        self.coordination = coordination
        self.kinetics = kinetics
        self.name = "simple_relations" if kinetics else "motion_coordination" if coordination else "motion_independent"

    def predict(self, bundle):
        x = bundle.weights @ bundle.features
        sides = []
        for i in range(2):
            v = hand_target_features(bundle, i)
            p = np.full(7, .01)
            p[0] = .88
            # Valid approach toward a nearby object; neither contact nor load mandatory.
            if v is not None and v[5] > .5 and v[6] > .5 and v[8] > .5 and v[1]*.2 < -.004 and v[0]*.1 < .18:
                p[0], p[1] = .08, .81
            # Load alone deliberately does not create a new request.
            if self.kinetics and v is not None and v[4] > .5 and v[3] > .5 and v[2] > .5 and p[1] > .5:
                p[1], p[4] = .2, .61  # illustrative NT alternative, not a validated physical rule
            sides.append(p/p.sum())
        joint = np.outer(*sides)
        if self.coordination and x[32] > .3:
            # Modest joint approach preference; not a physical gate or fixed dominance.
            joint[1, 1] *= 1.5
        return (joint/joint.sum()).ravel()


class UnresolvedBoundary:
    name = "unresolved_A"
    def predict(self, bundle):
        raise NotImplementedError("external-method equations/configuration unresolved; no invented reproduction")


# ---- per-action descriptive floor on the cue block ---------------------------------------------------
# Declared, untuned thresholds in the units of the cue block (metres, m/s, radians, cosines), re-expressed from the
# reference verified cue definitions (index direction, reversals, held-over-target, swept angle,
# lowering, stillness); like ``SimpleRules`` a transparent floor, never a fitted model and not a external method
# reconstruction: no Boltzmann likelihood, no recursion, no beta.
ACTION_RULE_THRESHOLDS = dict(hold_inplane_m=.07, hold_object_height_m=.03, still_rate_mps=.05, lowering_rate_mps=-.04,
                              release_height_m=.16, reversals_min=2, sweep_window_rad=.6, over_inplane_m=.08,
                              over_height_m=.02, index_alignment_min=.9, press_distance_m=.15)


class ActionRules(SimpleRules):
    """``SimpleRules`` for grasp plus one declared rule per other action, read from the cue block of the most-supported
    object; order of precedence when several fire: R, NT, AR, HP, FP, then G. Under layout v1 (no cues) it equals
    ``SimpleRules`` exactly."""
    def __init__(self, coordination=False):
        super().__init__(coordination=coordination)
        self.name = "motion_actions_floor" + ("_coordination" if coordination else "")
        self.thresholds = dict(ACTION_RULE_THRESHOLDS)

    def predict(self, bundle):
        base = super().predict(bundle).reshape(7, 7)
        if layout_of(bundle.layout).cues_per_hand == 0:
            return base.ravel()
        marginals = [base.sum(axis=1), base.sum(axis=0)]
        sides = []
        for i in range(2):
            p = marginals[i].copy()
            c = hand_action_cues(bundle, i); v = hand_target_features(bundle, i)
            action = self._action(c, v) if c is not None else None
            if action is not None:
                p[:] = .01; p[0] = .08; p[OUTCOMES.index(action)] = .81
            sides.append(p / p.sum())
        joint = np.outer(*sides)
        if self.coordination and (bundle.weights @ bundle.features)[32] > .3:
            joint[1, 1] *= 1.5
        return (joint / joint.sum()).ravel()

    @staticmethod
    def sweep_window(c):
        """The cue block carries the per-tick sweep rate; the window sum is accumulated by the rules themselves."""
        return abs(c[7] * 2.) * .25  # one tick's rate times a nominal 0.25 s window: declared, untuned

    def _action(self, c, v):
        t = self.thresholds
        holding = (c[6] * .1 < t["hold_inplane_m"] and c[4] * .2 > t["hold_object_height_m"] and c[2] > .5 and v is not None and v[6] > .5)
        hand_still = abs(c[1] * .2) < t["still_rate_mps"]
        if holding and c[5] * .2 < t["lowering_rate_mps"] and c[4] * .2 < t["release_height_m"]:
            return "R"
        if holding and c[3] * 4 >= t["reversals_min"]:
            return "NT"
        if holding and self.sweep_window(c) > t["sweep_window_rad"] and abs(c[5] * .2) < t["still_rate_mps"]:
            return "AR"
        if holding and c[13] > .5 and c[11] * .1 < t["over_inplane_m"] and c[12] * .2 > t["over_height_m"] and hand_still:
            return "HP"
        if (not holding and c[10] > .5 and c[8] > t["index_alignment_min"] and v is not None and v[0] * .1 < t["press_distance_m"]
                and v[5] > .5 and hand_still):
            return "FP"
        return None
