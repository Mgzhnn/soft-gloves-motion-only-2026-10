"""Causal sequence readouts (GRU / causal Transformer / temporal CNN) over the hypothesis features.

These are additional intention arms with the same inputs, labels, supervisor and evaluation as
the per-tick readouts in ``inference.py``. Each comes in two variants:

* ``retain_hypotheses=True``  — every tick's competing hypotheses are embedded by a shared MLP and
  weight-pooled BEFORE the sequence model (the KT-R / KT-SET idea, now with memory);
* ``retain_hypotheses=False`` — the weighted-mean feature vector is fed directly (no hypothesis set).

Set versus pooled isolates nonlinear encoding before versus after pooling; it does not
isolate every possible benefit of retaining alternatives. ``representation="flat"`` feeds the
same stable, full-information vector as the per-tick flat comparator (``inference.flat_features``).
Set and pooled have identical parameter counts; flat has a wider first layer and its count is
reported. None of the arms is claimed to be size-matched beyond that.

Causality: the model keeps a per-session buffer of the last ``window`` ticks and never sees a later
tick. That windowing is what makes every arm causal; the Transformer's mask additionally keeps
attention inside the window causal. Training windows are built from the recorded bundle stream in the
same way. Parameter counts are reported because with 2–3 people any of these can memorise; all arms
are trained under person-level splits (``training.py``).

PyTorch is imported lazily so the rest of the package (and its tests) run without it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .inference import FEATURE_MODES, FLAT_DIM, Bundle, validate_model_mode, label_indices, flat_features, flat_dim, layout_of
from .training import LABEL_RULE, bundles_for, validate_recorded_training, Example

KINDS = ("gru", "transformer", "tcn")
IN_DIM = 41  # drop the constant bias column; kinetic columns are zero in motion-only bundles


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


@dataclass(frozen=True)
class Window:
    bundles: tuple[Bundle, ...]   # oldest → newest, newest is the decision tick
    label: int | tuple[int, ...] | None
    participant_id: str
    session_id: str
    packet_id: str
    decision_s: float
    opportunity_id: str = ""
    provenance: str = ""
    # How the window was built; recorded training refuses windows that differ from the model/replay.
    window: int | None = None
    max_gap_s: float | None = None


_REPLAY_GAP = object()  # sentinel: use the replay's own history-reset gap (Config().max_velocity_gap_s)


def _pad_set(bundles, max_h, in_dim=IN_DIM):
    """(T, max_h, in_dim) features, (T, max_h) weights (zero-padded), (T,) valid-tick mask."""
    T = len(bundles)
    feats = np.zeros((T, max_h, in_dim), dtype=np.float32)
    weights = np.zeros((T, max_h), dtype=np.float32)
    for t, b in enumerate(bundles):
        h = len(b.hypotheses)
        if h > max_h:
            raise ValueError("hypothesis capacity exceeded; refusing silent truncation")
        feats[t, :h] = b.features[:h, :in_dim]
        w = b.weights[:h]
        weights[t, :h] = w / max(float(w.sum()), 1e-9)
    return feats, weights


def build_windows(bundle_stream, window: int, max_gap_s=_REPLAY_GAP):
    """Yield causal windows (length ≤ window) ending at every tick of one ordered session stream.

    By default history is cleared at the same decision gap at which ``Replay.run`` resets the model,
    so training windows match runtime history. Pass ``max_gap_s=None`` only to disable that on purpose.
    """
    if type(window) is not int or window < 1:
        raise ValueError("positive window required")
    if max_gap_s is _REPLAY_GAP:
        from .replay import Config
        max_gap_s = Config().max_velocity_gap_s
    buf = deque(maxlen=window)
    previous = None
    for packet, bundle in bundle_stream:
        if previous is not None and max_gap_s is not None and packet.decision_s - previous > max_gap_s:
            buf.clear()
        previous = packet.decision_s
        buf.append(bundle)
        yield packet, tuple(buf)


def sequence_examples(packets, opportunities, window: int, config=None, scenes=None):
    """Windows ending at labelled ticks (same label rule as ``training.label_ticks``)."""
    from .training import label_ticks
    from .replay import Config
    config = config or Config()
    packets = tuple(packets)
    labelled, counts = label_ticks(packets, opportunities, config, scenes)
    by_packet = {e.packet_id: e for e in labelled}
    out = []
    for packet, bundles in build_windows(bundles_for(packets, config, scenes), window, config.max_velocity_gap_s):
        e = by_packet.get(packet.packet_id)
        if e is None:
            continue
        out.append(Window(bundles, e.label, e.participant_id, e.session_id, e.packet_id, e.decision_s,
                          e.opportunity_id, e.provenance, window, config.max_velocity_gap_s))
    return tuple(out), counts


class SequenceReadout:
    """Stateful causal readout; construct one per session (or call ``reset``)."""

    def __init__(self, kind="gru", retain_hypotheses=True, window=15, hidden=16, max_hypotheses=19,
                 layers=1, heads=2, seed=0, representation=None, layout="v1"):
        if kind not in KINDS:
            raise ValueError("unknown sequence model kind")
        if any(type(v) is not int or v < 1 for v in (window, hidden, max_hypotheses, layers, heads)):
            raise ValueError("positive integer sequence dimensions required")
        if kind == "transformer" and hidden % heads:
            raise ValueError("Transformer hidden dimension must divide evenly into heads")
        representation = representation or ("set" if retain_hypotheses else "pooled")
        if representation not in ("set", "pooled", "flat"):
            raise ValueError("unknown representation")
        torch, nn = _torch()
        torch.manual_seed(seed)
        self.representation = representation; self.seed = seed
        self.layout = layout_of(layout).name
        self.in_dim = layout_of(layout).dim - 1          # the bias column is dropped (v1: 41)
        self.flat_dim = flat_dim(layout)
        self.kind, self.retain, self.window, self.hidden, self.max_h = kind, retain_hypotheses, window, hidden, max_hypotheses
        self.name = f"seq_{kind}_{representation}"
        self.fit_provenance = "untrained"
        self.training_feature_mode = None
        self._buf = deque(maxlen=window)
        d = hidden
        self.embed = nn.Sequential(nn.Linear(self.flat_dim if representation == "flat" else self.in_dim, d), nn.Tanh())
        if kind == "gru":
            self.core = nn.GRU(d, d, num_layers=layers, batch_first=True)
        elif kind == "transformer":
            layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True)
            self.core = nn.TransformerEncoder(layer, num_layers=layers)
            self.pos = nn.Parameter(torch.zeros(1, window, d))
        else:
            blocks = []
            dilation = 1
            self.receptive_field = 1
            while self.receptive_field < window:
                blocks.extend([nn.ConstantPad1d((2*dilation, 0), 0),
                               nn.Conv1d(d, d, kernel_size=3, dilation=dilation), nn.Tanh()])
                self.receptive_field += 2*dilation
                dilation *= 2
            self.core = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.head = nn.Linear(d, 49)
        self._modules = [self.embed, self.core, self.head] + ([self.pos] if kind == "transformer" else [])
        self._torch, self._nn = torch, nn

    # ---- parameters -------------------------------------------------------------------------
    def parameters(self):
        for m in self._modules:
            if isinstance(m, self._nn.Parameter):
                yield m
            else:
                yield from m.parameters()

    def n_parameters(self):
        return int(sum(p.numel() for p in self.parameters()))

    def get_state(self):
        return [p.detach().clone() for p in self.parameters()]

    def set_state(self, state):
        with self._torch.no_grad():
            for p, value in zip(self.parameters(), state):
                p.copy_(value)

    # ---- forward ----------------------------------------------------------------------------
    def _forward(self, feats, weights, lengths):
        """feats (B, T, H, 41), weights (B, T, H), lengths (B,) → (B, 49) probabilities for the last tick."""
        return self._torch.softmax(self._logits(feats, weights, lengths), dim=-1)

    def _log_likelihood(self, feats, weights, lengths, targets):
        """log P(any allowed joint outcome), computed in log space so no probability is clamped.

        Equals log((softmax * targets).sum()) wherever that is representable, but keeps a gradient
        for confidently wrong examples instead of zeroing it at a 1e-12 floor.
        """
        torch = self._torch
        logp = torch.log_softmax(self._logits(feats, weights, lengths), dim=-1)
        return torch.logsumexp(logp.masked_fill(targets <= 0, float("-inf")), dim=1)

    def _logits(self, feats, weights, lengths):
        torch = self._torch
        B, T = feats.shape[:2]
        if self.representation == "set":
            x = (self.embed(feats) * weights.unsqueeze(-1)).sum(dim=2)
        elif self.representation == "flat":
            x = self.embed(feats)  # (B, T, FLAT_DIM): the stable full-information vector per tick
        else:
            x = self.embed((feats * weights.unsqueeze(-1)).sum(dim=2))
        # Run only real ticks. Padding must not advance GRU state or positional embeddings.
        last = torch.zeros(B, self.hidden, dtype=x.dtype, device=x.device)
        for length in torch.unique(lengths).tolist():
            idx = torch.nonzero(lengths == length).flatten()
            actual = x[idx, :length]
            if self.kind == "gru":
                out, _ = self.core(actual)
                result = out[:, -1]
            elif self.kind == "transformer":
                causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=x.device), diagonal=1)
                result = self.core(actual + self.pos[:, :length], mask=causal)[:, -1]
            else:
                result = self.core(actual.transpose(1, 2))[:, :, -1]
            last = last.index_copy(0, idx, result)
        return self.head(last)

    def _tensors(self, windows):
        torch = self._torch
        T = self.window
        flat = self.representation == "flat"
        feats = np.zeros((len(windows), T, self.flat_dim) if flat else (len(windows), T, self.max_h, self.in_dim), dtype=np.float32)
        weights = np.zeros((len(windows), T, 1 if flat else self.max_h), dtype=np.float32)
        lengths = np.zeros(len(windows), dtype=np.int64)
        for i, w in enumerate(windows):
            if not w:
                raise ValueError("empty causal window")
            for bundle in w:
                validate_model_mode(self, bundle)
                if len(bundle.hypotheses) > self.max_h:
                    raise ValueError("hypothesis capacity exceeded; refusing silent truncation")
            if flat:
                f = np.stack([flat_features(b) for b in w[-T:]]); n = f.shape[0]
                feats[i, :n] = f
            else:
                f, wt = _pad_set(w[-T:], self.max_h, self.in_dim)
                n = f.shape[0]
                feats[i, :n] = f; weights[i, :n] = wt
            lengths[i] = n
        return torch.from_numpy(feats), torch.from_numpy(weights), torch.from_numpy(lengths)

    # ---- runtime interface (same as the per-tick readouts) ----------------------------------
    def reset(self):
        self._buf.clear()

    def predict(self, bundle):
        validate_model_mode(self, bundle)
        window = tuple(self._buf) + (bundle,)
        f, w, n = self._tensors([window[-self.window:]])  # raises before the tick enters the history
        self._buf.append(bundle)
        with self._torch.no_grad():
            p = self._forward(f, w, n)[0].cpu().numpy().astype(float)
        return p / p.sum()

    def predict_window(self, bundles):
        """Stateless prediction for an explicit causal window (used for training/eval checks)."""
        f, w, n = self._tensors([tuple(bundles)])
        with self._torch.no_grad():
            p = self._forward(f, w, n)[0].cpu().numpy().astype(float)
        return p / p.sum()

    # ---- training ---------------------------------------------------------------------------
    def _fit(self, windows, labels, steps=200, lr=1e-2, regularization=1e-4, batch=64, callback=None):
        """``steps`` = passes over all windows (as for the full-batch per-tick arms), in shuffled
        minibatches whose order depends on the model seed. ``callback(pass_number)`` runs after each pass."""
        torch = self._torch
        f, w, n = self._tensors([win for win in windows])
        if not windows or len(windows) != len(labels):
            raise ValueError("aligned nonempty windows and labels required")
        targets = np.zeros((len(labels), 49), dtype=np.float32)
        for i, label in enumerate(labels):
            targets[i, label_indices(label)] = 1.
        y = torch.from_numpy(targets)
        if steps < 1 or lr <= 0 or regularization < 0:
            raise ValueError("invalid fit settings")
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=regularization)
        self.fit_diagnostics = dict(optimizer="Adam", loss_curve=[])
        def checkpoint(step):
            with torch.no_grad():
                loss = float(-self._log_likelihood(f,w,n,y).mean())
            if not np.isfinite(loss):
                raise ValueError("nonfinite fitting loss")
            self.fit_diagnostics["loss_curve"].append(dict(step=step,unregularized_log_loss=loss))
        checkpoint(0)
        self.training_feature_mode = windows[0][-1].feature_mode
        g = torch.Generator().manual_seed(int(self.seed))
        for step in range(steps):
            for idx in torch.randperm(len(windows), generator=g).split(batch):
                loss = -self._log_likelihood(f[idx], w[idx], n[idx], y[idx]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            if (step+1) % max(1,steps//10) == 0 or step+1 == steps:
                checkpoint(step+1)
            if callback is not None and callback(step+1):
                break  # the caller's early-stopping rule (development patience) asked to stop

    def fit_synthetic(self, windows, labels, steps=100, lr=1e-2, regularization=1e-4):
        if any(b.source_kinds != ("synthetic",) for win in windows for b in win):
            raise ValueError("synthetic trainer rejects recorded/measured source bundles")
        modes = {b.feature_mode for win in windows for b in win}
        if len(modes) != 1 or not modes <= set(FEATURE_MODES):
            raise ValueError("training requires one explicit feature mode")
        self._fit(windows, labels, steps, lr, regularization)
        self.fit_provenance = "synthetic_only_no_reference_validation"

    def fit_recorded(self, examples, split_id, steps=200, lr=1e-2, regularization=1e-4, callback=None):
        """``examples`` are ``Window`` objects with labels from the ledger (see ``sequence_examples``)."""
        if not examples or any(not isinstance(e, Window) or e.label is None for e in examples):
            raise ValueError("labelled recorded windows required")
        flat = [Example(b, e.label, e.participant_id, e.session_id, e.packet_id, e.decision_s, e.opportunity_id, e.provenance)
                for e in examples for b in e.bundles]
        validate_recorded_training(flat)
        if any(e.window is None or e.max_gap_s is None for e in examples):
            raise ValueError("recorded windows must declare their window length and gap rule; build them with sequence_examples")
        if any(e.window != self.window for e in examples):
            raise ValueError("training windows were built with a different window length than this model uses at runtime")
        gaps = {e.max_gap_s for e in examples}
        if len(gaps) != 1:
            raise ValueError("training windows mix different history-reset gaps")
        self.training_reset_gap_s = next(iter(gaps))
        self._fit([e.bundles for e in examples], [e.label for e in examples], steps, lr, regularization, callback=callback)
        people = sorted({e.participant_id for e in examples})
        self.fit_provenance = (f"recorded_ledger_labels|participants={','.join(people)}|n={len(examples)}"
                               f"|split={split_id}|rule={LABEL_RULE}|window={self.window}|reset_gap_s={gaps.pop()}"
                               f"|params={self.n_parameters()}")
        return self
