"""Simulated glove supervisor: a per-hand finite state machine that turns proposals into accepted glove commands.

Simulation only (16 October 2026). No hardware API is called here: the output is a list
of ``Command`` records that a glove interface would consume, and a per-tick state trace. The structure follows the
reference state machine (Hand Open -> Close Hand -> Tighten Finger -> Close Hand -> Hand Open; Hand Open -> Extend Index
-> Hand Open) and its execution conditions (contact-type actions execute when the hand is nearly still; motion-type
actions execute at once), with the safeguards a powered session needs and the reference did not state: a confirmation
count (the same proposal on consecutive ticks) against tick-scale flicker, a cool-down after every command, a stop
state, a veto input that opens the glove at once, and a hold rule: when the hand observation is lost the glove keeps its
state and never moves (an open glove stays open, a closed glove keeps holding) until a confirmed proposal or a veto.

Every duration below is a DECLARED placeholder, not a measurement: the glove's closure and opening times are unknown
(no measurement on 16 October 2026) and must be measured on the powered glove before any online use. The flexion targets are
the reference approximate stroke fractions (Extend Index ~0, Hand Open ~0.25, Close Hand ~0.55, Tighten ~0.8 of full
stroke, object-size adjusted) and are placeholders for the same reason.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from math import isfinite

from .contracts import SIDES, ACTIONS, OUTCOMES

STATES = ("open", "closing", "closed", "tightening", "tight", "extending_index", "releasing", "cooldown", "stopped")
COMMAND_KINDS = ("close", "tighten", "extend_index", "open", "stop")
# Which proposal drives which command from which state.
EDGES = {("open", "G"): "close", ("closed", "AR"): "tighten", ("closed", "NT"): "tighten", ("closed", "HP"): "tighten",
         ("tight", "AR"): "tighten", ("tight", "NT"): "tighten", ("tight", "HP"): "tighten",
         ("closed", "R"): "open", ("tight", "R"): "open", ("open", "FP"): "extend_index"}
CONTACT_TYPE = ("G", "FP", "HP", "R")   # execute when the hand is nearly still; AR and NT execute at once
COMMAND_OUTCOME = {"close": "G", "extend_index": "FP", "open": "R"}  # tighten carries its trigger (AR/NT/HP)
FLEXION_TARGET = dict(open=.25, close=.55, tighten=.8, extend_index=0., stop=None)  # reference , approximate; placeholders


@dataclass(frozen=True)
class FSMConfig:
    confirm_ticks: int = 2            # the same outcome proposed on this many consecutive ticks before it is acted on
    close_duration_s: float = .8      # placeholder: time from the close command to "closed" (to measure on the glove)
    open_duration_s: float = .8       # placeholder
    tighten_duration_s: float = .4    # placeholder
    extend_duration_s: float = .4     # placeholder
    done_timeout_s: float = 1.5       # tight -> closed when no tightening proposal is confirmed for this long
    extend_done_s: float = 1.5        # extending_index -> open after this long without a renewed FP proposal
    cooldown_s: float = 1.           # after any completed command no new command for this long
    execution_condition: str = "none" # "none" (the proposal's timing as is) or "still_contact" (contact-type actions wait for a still hand)
    still_rate_mps: float = .05       # "nearly still": |vertical rate| and |approach rate| below this (needs layout v2 cues)
    hold_on_observation_loss: bool = True  # lost hand observation: keep the state, issue nothing
    veto_opens: bool = True           # a veto at any state opens the glove at once (then cooldown)
    size_scaling: bool = True         # record the candidate object's footprint for force scaling when the trace carries it

    def __post_init__(self):
        if type(self.confirm_ticks) is not int or self.confirm_ticks < 1:
            raise ValueError("confirm_ticks must be a positive integer")
        for name in ("close_duration_s", "open_duration_s", "tighten_duration_s", "extend_duration_s", "done_timeout_s", "extend_done_s", "cooldown_s", "still_rate_mps"):
            v = getattr(self, name)
            if not isfinite(v) or v < 0:
                raise ValueError(f"{name} must be a finite nonnegative number of seconds")
        if self.execution_condition not in ("none", "still_contact"):
            raise ValueError("execution_condition must be 'none' or 'still_contact'")


@dataclass(frozen=True)
class Command:
    side: str
    t_s: float
    kind: str
    from_state: str
    to_state: str
    trigger_outcome: str | None
    trigger_score: float | None
    reason: str
    flexion_target_fraction: float | None
    object_size_m: float | None = None
    source: str = "simulation_only_no_hardware"

    def __post_init__(self):
        if self.side not in SIDES or self.kind not in COMMAND_KINDS or self.from_state not in STATES or self.to_state not in STATES:
            raise ValueError("invalid command")


@dataclass
class HandState:
    state: str = "open"
    since_s: float = 0.
    streak_outcome: str | None = None
    streak: int = 0
    last_command_s: float | None = None
    last_confirmed_tighten_s: float | None = None
    last_fp_s: float | None = None
    blocked_reasons: dict = field(default_factory=dict)


class GloveStateMachine:
    """One machine per hand; ``step`` consumes one tick of the common supervisor's output."""

    def __init__(self, side, config: FSMConfig = FSMConfig()):
        if side not in SIDES:
            raise ValueError("side must be left or right")
        self.side, self.config, self.hand = side, config, HandState()

    def _block(self, reason):
        self.hand.blocked_reasons[reason] = self.hand.blocked_reasons.get(reason, 0) + 1

    def step(self, t, output, *, hand_observed=True, still=None, veto=False, stop=False, object_size_m=None):
        """Returns the Command issued at this tick or None. ``output`` is one entry of a trace row's ``outputs``."""
        h, c = self.hand, self.config
        commands = []
        # 1. hard stop and veto: independent of proposals and timing
        if stop and h.state != "stopped":
            commands.append(Command(self.side, t, "stop", h.state, "stopped", None, None, "external_stop", FLEXION_TARGET["stop"]))
            h.state, h.since_s, h.last_command_s = "stopped", t, t; h.streak = 0
            return commands[0]
        if h.state == "stopped":
            self._block("stopped_until_reset"); return None
        if veto and c.veto_opens and h.state not in ("open", "releasing"):
            commands.append(Command(self.side, t, "open", h.state, "releasing", None, None, "veto", FLEXION_TARGET["open"]))
            h.state, h.since_s, h.last_command_s = "releasing", t, t; h.streak = 0
            return commands[0]
        # 2. timed transitions of transitional states
        elapsed = t - h.since_s + 1e-9  # a tick exactly at the duration counts (0.3 - 0.1 < 0.2 in floating point)
        if h.state == "closing" and elapsed >= c.close_duration_s:
            h.state, h.since_s = "closed", t
        elif h.state == "releasing" and elapsed >= c.open_duration_s:
            h.state, h.since_s = "cooldown", t
        elif h.state == "tightening" and elapsed >= c.tighten_duration_s:
            h.state, h.since_s = "tight", t
        elif h.state == "extending_index" and elapsed >= c.extend_duration_s:
            h.state, h.since_s = "extending_index", h.since_s  # stays until done
        if h.state == "cooldown" and elapsed >= c.cooldown_s:
            h.state, h.since_s = "open", t
        if h.state == "tight" and h.last_confirmed_tighten_s is not None and t - h.last_confirmed_tighten_s + 1e-9 >= c.done_timeout_s:
            h.state, h.since_s = "closed", t  # the reference Done edge: back to holding
        if h.state == "extending_index" and h.last_fp_s is not None and t - h.last_fp_s + 1e-9 >= c.extend_done_s and elapsed >= c.extend_duration_s:
            h.state, h.since_s = "open", t
        # 3. proposal confirmation (debounce)
        proposed = output.get("status") == "proposed_simulation_only"
        outcome = output.get("raw_outcome") if proposed else None
        if not hand_observed and c.hold_on_observation_loss:
            self._block("hand_observation_lost_hold_state"); h.streak = 0; h.streak_outcome = None
            return None
        if outcome is None:
            h.streak, h.streak_outcome = 0, None
            return None
        if outcome == h.streak_outcome:
            h.streak += 1
        else:
            h.streak_outcome, h.streak = outcome, 1
        if h.streak < c.confirm_ticks:
            self._block("awaiting_confirmation"); return None
        if outcome in ("AR", "NT", "HP"):
            h.last_confirmed_tighten_s = t
        if outcome == "FP":
            h.last_fp_s = t
        # 4. the edge from the current state, with the cool-down and the execution condition
        kind = EDGES.get((h.state, outcome))
        if kind is None:
            self._block(f"no_edge_from_{h.state}_on_{outcome}"); return None
        if h.last_command_s is not None and t - h.last_command_s + 1e-9 < c.cooldown_s and h.state in ("open", "closed", "tight"):
            # cool-down after the previous command; the tight state renews on its own confirmed proposals
            if not (h.state == "tight" and kind == "tighten"):
                self._block("cooldown_after_previous_command"); return None
        if c.execution_condition == "still_contact" and outcome in CONTACT_TYPE:
            if still is None:
                self._block("stillness_unobservable_without_scene_cues"); return None
            if not still:
                self._block("execution_condition_hand_moving"); return None
        if h.state == "tight" and kind == "tighten":
            return None  # already tight: the confirmed proposal only renews the Done timer
        to_state = dict(close="closing", tighten="tightening", extend_index="extending_index", open="releasing")[kind]
        target = FLEXION_TARGET[kind]
        if kind in ("close", "tighten") and c.size_scaling and object_size_m is not None and isfinite(object_size_m):
            # the reference object-size adjustment: smaller objects pull further; +-0.1 band, declared placeholder
            target = min(1., max(0., target + .1 * (1. - min(max(object_size_m, .03), .13) / .08)))
        command = Command(self.side, t, kind, h.state, to_state, outcome, float(output.get("outcome_score") or 0.),
                          "confirmed_proposal_" + ("executed_at_once" if outcome not in CONTACT_TYPE else "contact_type"),
                          target, object_size_m)
        h.state, h.since_s, h.last_command_s, h.streak = to_state, t, t, 0
        return command


def _still(row, i, threshold=None):
    """Nearly still from the trace's scene record (vertical rate of the hand); None when unobservable (layout v1).

    The approach rate is not stored in a trace row, so stillness here is the plane-anchored vertical rate alone; the
    reference condition (hand velocity near zero) is approximated by that one camera-invariant component."""
    scene = row.get("scene")
    if not scene or not scene.get("plane_available"):
        return None
    rate = scene.get("height_rates", {}).get(SIDES[i])
    if rate is None:
        return None
    return abs(rate) < (FSMConfig().still_rate_mps if threshold is None else threshold)


def simulate_commands(rows, config: FSMConfig = FSMConfig(), vetoes=(), stops=()):
    """Run the two machines over one session's trace rows (time-ordered). Returns (commands, states, summary).

    ``vetoes`` / ``stops``: iterables of (side, t_s) external events, applied at the first tick at or after t_s.
    ``states``: per tick ``{"t_s", "left", "right"}``; ``summary`` counts commands and blocked reasons per hand.
    """
    machines = {side: GloveStateMachine(side, config) for side in SIDES}
    pending_veto = {side: sorted(t for s, t in vetoes if s == side) for side in SIDES}
    pending_stop = {side: sorted(t for s, t in stops if s == side) for side in SIDES}
    commands, states = [], []
    last = -1.
    for row in rows:
        t = row["decision_s"]
        if t <= last:
            raise ValueError("time-ordered trace rows required")
        last = t
        tick = dict(t_s=t, packet_id=row.get("packet_id"))
        for i, side in enumerate(SIDES):
            out = row["outputs"][i]
            observed = row["inputs"]["hands"][i].get("xyz") is not None
            veto = bool(pending_veto[side]) and pending_veto[side][0] <= t
            stop = bool(pending_stop[side]) and pending_stop[side][0] <= t
            if veto:
                pending_veto[side].pop(0)
            if stop:
                pending_stop[side].pop(0)
            command = machines[side].step(t, out, hand_observed=observed, still=_still(row, i, config.still_rate_mps), veto=veto, stop=stop)
            if command is not None:
                commands.append(command)
            tick[side] = machines[side].hand.state
        states.append(tick)
    summary = {side: dict(commands={k: sum(1 for c in commands if c.side == side and c.kind == k) for k in COMMAND_KINDS},
                          blocked=dict(machines[side].hand.blocked_reasons), final_state=machines[side].hand.state) for side in SIDES}
    summary["scope"] = "simulated glove supervisor over an offline trace; no hardware, no measured glove timing; durations are declared placeholders"
    summary["config"] = asdict(config)
    return commands, states, summary


def command_rows(rows, commands):
    """Trace rows rewritten so that a hand 'proposes' exactly at the ticks where the supervisor ACCEPTED a command.

    The outcome of a tick is the command's action (close -> G, open -> R, extend_index -> FP, tighten -> its trigger),
    so ``evaluation.evaluate_opportunities`` scores accepted commands with the same first-event rule as proposals and
    ``stats.unwanted_exposure`` counts unwanted COMMANDS per labelled no-request hand-minute. A stop issues nothing.
    """
    by_tick = {}
    for c in commands:
        if c.kind == "stop":
            continue
        by_tick[(c.side, c.t_s)] = COMMAND_OUTCOME.get(c.kind, c.trigger_outcome)
    out = []
    for row in rows:
        outputs = []
        for i, side in enumerate(SIDES):
            original = row["outputs"][i]
            action = by_tick.get((side, row["decision_s"]))
            if action is None:
                outputs.append(dict(original, status="no_new_request" if original["status"] != "withheld" else "withheld",
                                    raw_outcome="NQ" if original["status"] != "withheld" else original.get("raw_outcome", "NQ"),
                                    accepted_command=None))
            else:
                outputs.append(dict(original, status="proposed_simulation_only", raw_outcome=action, accepted_command=action))
        out.append(dict(row, outputs=outputs, endpoint="accepted_command_secondary"))
    return out


def commands_record(commands):
    return [asdict(c) for c in commands]
