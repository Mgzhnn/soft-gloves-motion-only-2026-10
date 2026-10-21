"""Auditable offline person-held-out comparison. No labels enter inference or supervisor."""
from dataclasses import asdict, replace
import hashlib
import json
import numpy as np
from .evaluation import evaluate_opportunities, timeline_accounting
from .inference import ActionRules, FactorizedJoint, FlatJoint, HardCollapseLinear, HypothesisPoolMLP, LinearJoint, SimpleRules
from .replay import Config, Replay
from .stats import MIN_PEOPLE_FOR_INTERVAL, per_participant_success, unwanted_exposure, paired_contrast, paired_interval
from .training import label_ticks, fit_recorded, mean_log_loss, development_split, session_development_split

SEQUENCE_KINDS=("gru","transformer","tcn")
# "gru" etc. keep their historical meaning (set representation); *_pooled / *_flat are the matched pairs.
SEQUENCE_ARMS=SEQUENCE_KINDS+tuple(f"{k}_{r}" for k in SEQUENCE_KINDS for r in ("pooled","flat"))
RULE_ARMS=("rules","rules_mc")  # hand-authored, never fitted; since 2 Oct 2026 a descriptive floor (they decide identically at every grid threshold)
FLOOR_ARMS=RULE_ARMS+("rules_actions",)  # every unfitted arm; rules_actions (16 Oct 2026) is the per-action floor on the cue block (layout v2), outside CORE_ARMS
SIMPLE_ARMS=("mi_learned","mc_learned")  #'s M-I / M-C as models fitted on the same labels and budget (2 Oct 2026)
COLLAPSE_ARMS=("kt_r_mean","kt_r_hard")  # KT-R's own readout on the collapsed hypothesis set: the mechanism controls
CORE_ARMS=RULE_ARMS+SIMPLE_ARMS+("flat","kt_set","kt_r")+COLLAPSE_ARMS
ARMS=CORE_ARMS+("rules_actions",)+SEQUENCE_ARMS
STRONGEST={"strongest_simple":SIMPLE_ARMS,"strongest_rule":RULE_ARMS}  # controls chosen per person on development success; tie -> the first (simpler) arm


def factory(name,seed=0,window=15,layout="v1"):
    """One arm; ``layout`` (16 Oct 2026) sizes the learned readouts for the feature layout of the replay configuration."""
    if name=="rules":return SimpleRules()
    if name=="rules_mc":return SimpleRules(coordination=True)
    if name=="rules_actions":return ActionRules()
    if name=="mi_learned":return FactorizedJoint(layout=layout)
    if name=="mc_learned":return FactorizedJoint(cross_hand=True,coupling=True,layout=layout)
    if name=="flat":return FlatJoint(seed,layout=layout)
    if name=="kt_set":return HypothesisPoolMLP(seed,layout=layout)
    if name=="kt_r":return LinearJoint(layout=layout)
    if name=="kt_r_mean":return LinearJoint(mixture=False,layout=layout)
    if name=="kt_r_hard":return HardCollapseLinear(layout=layout)
    if name in SEQUENCE_ARMS:
        from .sequence import SequenceReadout
        kind,_,representation=name.partition("_")
        return SequenceReadout(kind,window=window,seed=seed,representation=representation or "set",layout=layout)
    raise ValueError("unknown arm")


def parameter_count(model):
    if hasattr(model,"n_parameters"):return model.n_parameters()
    return sum(getattr(model,k).size for k in ("W","A","a","B","b") if hasattr(model,k))


def validate_sessions(sessions):
    seen=set()
    for session in sessions:
        packets=session["packets"];ops=session["opportunities"]
        if not packets:raise ValueError("empty packet stream")
        identity=(packets[0].participant_id,packets[0].session_id)
        if identity in seen:raise ValueError("repeated participant/session")
        seen.add(identity)
        if any((p.participant_id,p.session_id)!=identity for p in packets):raise ValueError("mixed session packets")
        if any((o.participant_id,o.session_id)!=identity for o in ops):raise ValueError("ledger identity mismatch")
        ordered=sorted(ops,key=lambda o:o.start_s)
        if any(a.end_s>b.start_s for a,b in zip(ordered,ordered[1:])):raise ValueError("overlapping reference intervals")
    if len({p for p,s in seen})<2:raise ValueError("person-held-out comparison needs at least two people; use diagnostic replay for one operator")


def cadence(packets):
    """Decision-tick spacing actually replayed; temporal windows count ticks, so this must be reported."""
    gaps=np.diff([p.decision_s for p in packets])
    return dict(ticks=len(packets),median_tick_interval_s=float(np.median(gaps)) if len(gaps) else None,
                max_tick_interval_s=float(gaps.max()) if len(gaps) else None)


def score(model,sessions,config,include_outcomes=False,early_limit_s=0.):
    outcomes=[];ops=[];coverage={};ticks={}
    for session in sessions:
        if not session["packets"]:raise ValueError("empty packet stream")
        # The join is by participant/session id and by time. A ledger imported under another id, or converted with a
        # wrong origin, used to join nothing: every request "missed", every no-request interval "clean", burden 0.0
        # and no error (architecture review, 5 October 2026). Such a session is refused here, on every scoring path.
        identity=(session["packets"][0].participant_id,session["packets"][0].session_id)
        if any((o.participant_id,o.session_id)!=identity for o in session["opportunities"]):raise ValueError("ledger identity mismatch")
        rows=Replay(config,model,scenes=session.get("scenes")).run(session["packets"])
        accounting=timeline_accounting(rows,session["opportunities"])
        if session["opportunities"] and accounting.get("unlabelled_hand_ticks",0)==accounting.get("all_hand_ticks",0):
            raise ValueError("no decision tick falls inside any labelled interval of session "+identity[1]+": check the ledger's origin_s and clock against the recording")
        outcomes.extend(evaluate_opportunities(rows,session["opportunities"],early_limit_s=early_limit_s))
        ops.extend(session["opportunities"])
        coverage[session["packets"][0].session_id]=accounting
        ticks[session["packets"][0].session_id]=cadence(session["packets"])
    result=dict(success=per_participant_success(outcomes),unwanted=unwanted_exposure(outcomes,ops),coverage=coverage,cadence=ticks)
    if include_outcomes:
        result["outcomes"]=outcomes  # per opportunity and hand: first proposal time, result, repeated proposals
    return result


def _get_state(model):
    if hasattr(model,"get_state"):return model.get_state()
    return {k:getattr(model,k).copy() for k in ("W","A","a","B","b") if hasattr(model,k)}


def _set_state(model,state):
    if hasattr(model,"set_state"):model.set_state(state);return
    for k,v in state.items():setattr(model,k,v.copy())


def _examples(model,sessions,window,config):
    out=[];accounting=[]
    for session in sessions:
        if hasattr(model,"predict_window"):
            from .sequence import sequence_examples
            ex,c=sequence_examples(session["packets"],session["opportunities"],window,config,session.get("scenes"))
        else:ex,c=label_ticks(session["packets"],session["opportunities"],config,session.get("scenes"))
        out.extend(ex);accounting.append(c)
    return out,accounting


def _proposes(metrics):
    """True when the arm proposed at least once on these sessions (request or no-request hands)."""
    return any(h.get("proposal_count",0)>0 for o in metrics.get("outcomes",[]) for h in o["hands"])

BURDEN_KEYS={"onsets":"onsets_per_hand_minute","ticks":"proposals_per_hand_minute"}
# The declared threshold grid: a true operating point on the best action's score, dense where
# rare-event probabilities live. The 23 Sep grid (.3-.9) is kept in older records.
THRESHOLD_GRID=(.02,.05,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9)


def fit_learned_arm(model,examples,dev_examples,steps,lr,regularization,split_id,synthetic_check=False,patience=None):
    """Fit one learned arm the way every comparison fold does; shared with the one-person diagnostic (16 Oct 2026).

    ``dev_examples``: labelled development examples for early stopping (the pass with the lowest development log loss
    is kept); ``[]`` when the development split has no labelled ticks (trained for all passes); ``None`` for a fixed
    protocol without a development split. ``patience``: stop after this many passes without a
    new best development loss (``None`` = train for all ``steps`` passes). Returns ``(loss_before, loss_after,
    stopping)``, both losses on the training examples, with the model left at the restored best pass when early
    stopping applied.
    """
    if patience is not None and (type(patience) is not int or patience<1):raise ValueError("patience must be a positive integer or None")
    before=mean_log_loss(model,examples)
    callback=None;stopping=None;track=None
    if dev_examples is None:
        stopping=dict(criterion=None,reason="fixed protocol without a development split; trained for all passes")
    elif dev_examples:
        track=dict(curve=[],best=None,best_pass=None,state=None)
        def callback(pass_number,model=model,dev_examples=dev_examples,track=track):
            loss=mean_log_loss(model,dev_examples);track["curve"].append(loss)
            if track["best"] is None or loss<track["best"]:
                track.update(best=loss,best_pass=pass_number,state=_get_state(model))
            return patience is not None and pass_number-track["best_pass"]>=patience  # True asks the fit loop to stop
    else:stopping=dict(criterion=None,reason="no labelled development ticks; trained for all passes")
    if synthetic_check:
        if hasattr(model,"predict_window"):
            model._fit([e.bundles for e in examples],[e.label for e in examples],steps,lr,regularization,callback=callback)
        else:model._fit([e.bundle for e in examples],[e.label for e in examples],steps,lr,regularization,callback=callback)
        model.fit_provenance="synthetic_experiment_software_check_only"
    elif hasattr(model,"predict_window"):
        model.fit_recorded(examples,split_id,steps,lr,regularization,callback=callback)
    else:fit_recorded(model,examples,split_id,steps,lr,regularization,callback=callback)
    if callback is not None:
        _set_state(model,track["state"])  # keep the pass with the lowest development loss
        passes=len(track["curve"])
        stopping=dict(criterion="development log loss",max_passes=steps,patience=patience,passes_run=passes,
                      stopped_by="patience" if passes<steps else "max_passes",best_pass=track["best_pass"],
                      development_loss_at_best=track["best"],development_loss_curve=track["curve"],
                      development_loss_after_restore=mean_log_loss(model,dev_examples))
    return before,mean_log_loss(model,examples),stopping


def risk_controlled_threshold(losses_by_person,thresholds,alpha):
    """Distribution-free selection of the operating point (16 Oct 2026, later): conformal risk control with a binary loss.

    ``losses_by_person[person][k]`` is 1 when that calibration person's burden at ``thresholds[k]`` exceeded the budget
    (or the point was silent for everyone), else 0; the loss is made monotone non-increasing in the threshold by taking
    the running maximum from the right (a higher threshold can only propose less, so this is conservative, never
    optimistic). The chosen point is the SMALLEST threshold whose conformal risk (sum of losses + 1) / (n + 1) is at most
    ``alpha`` (Angelopoulos et al., conformal risk control, loss bounded by 1; verify the citation before the paper).
    Under exchangeability of the calibration people and a new person, the expected loss of the new person is at most
    alpha: the budget holds for a new person with probability at least 1 - alpha. The smallest alpha that n people can
    certify is 1 / (n + 1). The threshold is None when no point meets alpha."""
    if not 0<alpha<1:raise ValueError("alpha must be in (0, 1)")
    people=sorted(losses_by_person);n=len(people)
    if n<1:raise ValueError("risk control needs at least one calibration person")
    K=len(thresholds)
    if any(len(losses_by_person[p])!=K for p in people):raise ValueError("one loss per threshold per person")
    mono={p:[max(losses_by_person[p][k:]) for k in range(K)] for p in people}
    risks=[(sum(mono[p][k] for p in people)+1)/(n+1) for k in range(K)]
    chosen=next((k for k in range(K) if risks[k]<=alpha),None)
    return dict(method="conformal risk control, binary loss (burden above budget, or silent for everyone), monotone envelope from the right",
                alpha=alpha,alpha_achievable=1/(n+1),n_calibration_people=n,calibration_people=people,
                conformal_risk_by_threshold=dict(zip(map(str,thresholds),risks)),
                raw_losses=dict(zip(map(str,thresholds),[[losses_by_person[p][k] for p in people] for k in range(K)])),
                threshold=(None if chosen is None else thresholds[chosen]),
                guarantee=("P(new person's burden <= budget) >= 1 - alpha under exchangeability; the operating point was chosen on "
                           "calibration people the model never saw; nothing was tuned on test people"))


def risk_control_selection(model,calibration,config,thresholds,budget,burden_unit,early_limit_s,alpha):
    """Out-of-sample burden curves of the calibration people -> binary losses -> ``risk_controlled_threshold``.
    Returns ``(selected, curve, feasible, reason, record)`` in the shape of ``development_selection`` so that the contrasts
    and reports read it unchanged; ``curve`` carries the calibration people's pooled metrics per threshold."""
    if burden_unit not in BURDEN_KEYS:raise ValueError("burden_unit must be 'onsets' or 'ticks'")
    if budget is None or budget<0:raise ValueError("declare the burden budget before risk control")
    burden_key=BURDEN_KEYS[burden_unit];curve=[];losses={}
    for threshold in thresholds:
        metrics=score(model,calibration,replace(config,proposal_score_threshold=threshold),include_outcomes=True,early_limit_s=early_limit_s)
        proposes=_proposes(metrics);metrics.pop("outcomes",None)
        for person,u in metrics["unwanted"].items():
            rate=u[burden_key]
            losses.setdefault(person,[]).append(1 if (rate is None or rate>budget) else 0)
        successes=[s["success_fraction"] for s in metrics["success"].values() if s["success_fraction"] is not None]
        curve.append(dict(threshold=threshold,feasible=None,within_budget=all(l[-1]==0 for l in losses.values()),proposes=proposes,
                          success=float(np.mean(successes)) if successes else None,metrics=metrics))
    missing=sorted({s["packets"][0].participant_id for s in calibration}-set(losses))
    if missing:raise ValueError(f"calibration people without labelled no-request exposure cannot calibrate a burden: {missing}")
    # a point silent on every calibration person is a loss of 1 for everyone: a controller that never acts is not a usable point
    for k,point in enumerate(curve):
        if not point["proposes"]:
            for person in losses:losses[person][k]=1
    record=risk_controlled_threshold(losses,tuple(thresholds),alpha)
    for point in curve:point["feasible"]=(point["threshold"]==record["threshold"])
    if record["threshold"] is None:
        return max(thresholds),curve,False,f"no operating point certifies the budget at alpha {alpha} on {record['n_calibration_people']} calibration people (achievable alpha {record['alpha_achievable']:.3f})",record
    return record["threshold"],curve,True,f"risk-controlled: smallest threshold whose conformal risk of exceeding the budget ({burden_unit} per hand-minute) is at most {alpha}",record


def development_selection(model,dev,config,thresholds,budget,burden_unit,early_limit_s=0.):
    """Choose the reporting threshold on the development sessions; shared with the one-person diagnostic (16 Oct 2026).

    The proposing operating point within the declared unwanted-proposal budget (``burden_unit`` per hand-minute) with
    the highest development success; a silent point is never selected; without a feasible point the largest grid
    threshold is reported with the reason. Returns ``(selected, dev_curve, feasible, reason)``.
    """
    if burden_unit not in BURDEN_KEYS:raise ValueError("burden_unit must be 'onsets' or 'ticks'")
    if budget is None or budget<0:raise ValueError("declare a development unwanted-proposal budget before selecting")
    burden_key=BURDEN_KEYS[burden_unit];dev_curve=[]
    for threshold in thresholds:
        metrics=score(model,dev,replace(config,proposal_score_threshold=threshold),include_outcomes=True,early_limit_s=early_limit_s)
        proposes=_proposes(metrics);metrics.pop("outcomes",None)
        rates=[r[burden_key] for r in metrics["unwanted"].values()]
        successes=[r["success_fraction"] for r in metrics["success"].values()]
        known=[s for s in successes if s is not None]
        within_budget=bool(rates) and all(r is not None and r<=budget for r in rates) and bool(known)
        dev_curve.append(dict(threshold=threshold,feasible=within_budget and proposes,within_budget=within_budget,proposes=proposes,
                              success=float(np.mean(known)) if known else None,metrics=metrics))
    choices=[p for p in dev_curve if p["feasible"]]
    if choices:
        selected=max(choices,key=lambda p:(p["success"],p["threshold"]))["threshold"]
        reason=f"development-selected: proposing operating point within the unwanted-proposal budget ({burden_unit} per hand-minute) with the highest development success"
    else:
        selected=max(thresholds)
        reason=("silent: only operating points that never proposed on development data met the budget"
                if any(p["within_budget"] for p in dev_curve) else
                "no operating point met the development unwanted-proposal budget")
    return selected,dev_curve,bool(choices),reason


def compare(sessions,arms=CORE_ARMS,seeds=(0,),steps=2000,lr=.01,
            regularization=.001,thresholds=THRESHOLD_GRID,fixed_threshold=.6,
            selection="fixed",max_unwanted_proposals_per_hand_minute=None,burden_unit="onsets",window=15,
            config=Config(),synthetic_check=False,outer_test_people=None,contrast="primary_contrast",
            early_limit_s=0.,min_people_for_interval=MIN_PEOPLE_FOR_INTERVAL,patience=None,
            risk_alpha=None,calibration_people=None):
    """Threshold curves are descriptive; reporting threshold is fixed or chosen on development.

    Development selection uses the declared unwanted-proposal budget then request success. A development
    operating point that never proposes anything is SILENT and is never selected (2 Oct 2026): a
    controller that never acts trivially meets any burden budget, and the old rule then compared KT-R
    with a control that could not succeed. If no proposing threshold meets the budget, the reporting
    point is explicitly infeasible (no test-data rescue) and that person is excluded from every contrast
    involving the arm, with the reason. With a development split, every learned arm is trained for up to
    ``steps`` passes over its training examples and keeps the pass with the lowest development log loss;
    the same rule applies to every arm. The primary contrast follows accepted unless another
    recorded ``contrast`` is named. ``early_limit_s`` is the declared early-proposal tolerance of the
    endpoint (0 = strict). ``burden_unit`` says what the development budget counts: ``"onsets"`` (proposal
    runs starting inside labelled no-request time, per hand-minute; the plan default accepted on 16 October
    2026,) or ``"ticks"`` (every proposal tick, the 23 September reading, kept for older records). This
    is a feasibility protocol, not calibration or evidence of the budget transferring to new people.
    ``selection="risk_control"`` (16 Oct 2026, later) replaces the development choice by conformal risk control: the
    training people of each outer fold are split into FIT people and the last ``calibration_people`` CALIBRATION people
    (default: half, rounded up, at least one); every learned arm is fitted on the fit people for all passes (no early
    stopping, so the calibration people stay unseen), each calibration person's out-of-sample burden curve becomes a
    binary loss against the budget, and ``risk_controlled_threshold`` picks the smallest threshold whose conformal risk
    is at most ``risk_alpha``: a new person's burden then stays within the budget with probability at least 1 - alpha
    under exchangeability. Success is read at that point, never used to choose it. With few people the certifiable
    alpha is coarse (1 / (n + 1)); the record states it.
    ``steps`` is the pass budget: the frozen rule fits every learned arm for all 2000 passes and
    keeps the pass with the lowest development log loss (``patience=None``). A ``patience`` value stops after that many
    passes without a new best; it is not frozen because 50 cut the set-pooling control off on a plateau that every seed
    leaves by pass 400. The fixed protocol has no development loss and runs ``steps`` passes. The default grid is ``THRESHOLD_GRID`` and the supervisor's
    ``best_action_threshold`` rule makes every grid point an operating point (``config.decision_rule``).
    """
    if contrast not in CONTRAST_SPECS:raise ValueError("unknown contrast specification")
    if patience is not None and (type(patience) is not int or patience<1):raise ValueError("patience must be a positive integer or None")
    if selection not in ("fixed","development","risk_control") or steps<1:raise ValueError("invalid fit protocol")
    if selection=="risk_control":
        if risk_alpha is None or not 0<risk_alpha<1:raise ValueError("risk control needs risk_alpha in (0, 1)")
        if calibration_people is not None and (type(calibration_people) is not int or calibration_people<1):raise ValueError("calibration_people must be a positive integer")
    if any(a not in ARMS for a in arms):raise ValueError("unknown model arm")
    if not thresholds or any(not 0<t<=1 for t in thresholds) or not 0<fixed_threshold<=1:raise ValueError("invalid threshold grid")
    if selection in ("development","risk_control") and (max_unwanted_proposals_per_hand_minute is None or max_unwanted_proposals_per_hand_minute < 0):
        raise ValueError("declare a development unwanted-proposal budget before fitting")
    if burden_unit not in ("onsets","ticks"):raise ValueError("burden_unit must be 'onsets' or 'ticks'")
    if type(min_people_for_interval) is not int or min_people_for_interval<2:
        raise ValueError("min_people_for_interval must be an integer of at least 2")  # checked before any fitting, not after it
    validate_sessions(sessions)
    people=sorted({s["packets"][0].participant_id for s in sessions});folds=[]
    if outer_test_people is not None and (not outer_test_people or not set(outer_test_people) <= set(people)):
        raise ValueError("invalid outer test people")
    for held in (people if outer_test_people is None else outer_test_people):
        outer_train=[s for s in sessions if s["packets"][0].participant_id!=held]
        test=[s for s in sessions if s["packets"][0].participant_id==held]
        dev=[];train=outer_train;calibration=[]
        if selection=="risk_control":
            train_people=sorted({s["packets"][0].participant_id for s in train})
            if len(train_people)<2:raise ValueError("risk control needs at least two training people (fit and calibration)")
            k=calibration_people if calibration_people is not None else max(1,(len(train_people)+1)//2)
            if k>=len(train_people):raise ValueError("calibration_people must leave at least one fit person")
            cal_people=set(train_people[-k:])
            calibration=[s for s in train if s["packets"][0].participant_id in cal_people]
            train=[s for s in train if s["packets"][0].participant_id not in cal_people]
        if selection=="development":
            train_people=sorted({s["packets"][0].participant_id for s in train})
            if len(train_people)>1:
                dev=[s for s in train if s["packets"][0].participant_id==train_people[-1]]
            elif len(train)>1:dev=[train[-1]]
            else:raise ValueError("no independent development person/session; use a preset fixed protocol or collect another session")
            dev_ids={(s["packets"][0].participant_id,s["packets"][0].session_id) for s in dev}
            train=[s for s in train if (s["packets"][0].participant_id,s["packets"][0].session_id) not in dev_ids]
        for name in arms:
            for seed in (seeds if name not in FLOOR_ARMS else seeds[:1]):
                model=factory(name,seed,window,config.feature_layout)
                examples,accounting=_examples(model,train,window,config)
                if not examples:raise ValueError("no usable training labels in fold")
                before=after=None;stopping=None
                if name not in FLOOR_ARMS:
                    # None = fixed protocol (no development split); [] = a development split without labelled ticks
                    dev_examples=_examples(model,dev,window,config)[0] if selection=="development" else None
                    before,after,stopping=fit_learned_arm(model,examples,dev_examples,steps,lr,regularization,f"outer:{held}",synthetic_check,patience)
                selected=fixed_threshold;dev_curve=[];feasible=True;selection_reason="fixed reporting threshold; no development selection";risk=None
                if selection=="development":
                    selected,dev_curve,feasible,selection_reason=development_selection(model,dev,config,thresholds,max_unwanted_proposals_per_hand_minute,burden_unit,early_limit_s)
                elif selection=="risk_control":
                    selected,dev_curve,feasible,selection_reason,risk=risk_control_selection(model,calibration,config,thresholds,max_unwanted_proposals_per_hand_minute,burden_unit,early_limit_s,risk_alpha)
                curve=[dict(threshold=t,metrics=score(model,test,replace(config,proposal_score_threshold=t),early_limit_s=early_limit_s)) for t in thresholds]
                result=dict(held_out_person=held,arm=name,seed=seed,parameter_count=parameter_count(model),
                    train_sessions=[(s["packets"][0].participant_id,s["packets"][0].session_id) for s in train],
                    development_sessions=[(s["packets"][0].participant_id,s["packets"][0].session_id) for s in dev],
                    steps=steps if name not in FLOOR_ARMS else 0,learning_rate=lr,loss_before=before,loss_after=after,
                    early_stopping=stopping,
                    fitting_diagnostics=getattr(model,"fit_diagnostics",None),
                    fitting_improved=None if after is None else after<before,training_label_accounting=accounting,
                    threshold_selection=selection,reporting_threshold=selected,development_budget_feasible=feasible,
                    development_selection_reason=selection_reason,
                    development_budget_unwanted_proposals_per_hand_minute=max_unwanted_proposals_per_hand_minute,
                    development_budget_unit=burden_unit,risk_control=risk,
                    calibration_sessions=[(s["packets"][0].participant_id,s["packets"][0].session_id) for s in calibration],
                    reporting_metrics=score(model,test,replace(config,proposal_score_threshold=selected),include_outcomes=True,early_limit_s=early_limit_s),
                    descriptive_test_curve=curve,development_curve=dev_curve,fit_provenance=getattr(model,"fit_provenance","unvalidated_rules"))
                folds.append(result)
    # The contrasts named by the recorded specification, paired by person and seed. Seeds are not people.
    contrasts=contrasts_from_folds(folds,seeds,contrast,min_people_for_interval)
    return dict(schema="sg-offline-comparison/3",scope="synthetic software check; participant N=0" if synthetic_check else "offline person-held-out feasibility comparison; intention compute latency excluded; no online assistance or efficacy claim",
                primary_contrast=CONTRAST_SPECS[contrast][0]["label"],contrast_specification=contrast,
                configuration=dict(window=window,steps=steps,patience=patience,lr=lr,regularization=regularization,thresholds=thresholds,seeds=seeds,
                                   selection=selection,max_unwanted_proposals_per_hand_minute=max_unwanted_proposals_per_hand_minute,burden_unit=burden_unit,
                                   risk_alpha=risk_alpha,calibration_people=calibration_people,
                                   early_limit_s=early_limit_s,min_people_for_interval=min_people_for_interval,
                                   fixed_threshold=fixed_threshold,rate_window=config.rate_window,
                                   replay_configuration=asdict(config),  # every replay setting, not only the rate window (5 Oct 2026)
                                   training="up to `steps` passes over the training examples for every learned arm; with a development split the pass with the lowest development log loss is kept and fitting stops after `patience` passes without a new best (None = all passes)",
                                   decision_rule=config.decision_rule,feature_layout=config.feature_layout,
                                   silent_points="a development operating point that never proposes is never selected and is reported as silent"),
                folds=folds,paired_contrasts=contrasts)


CONTRAST_SPECS={
    "primary_contrast":(
        dict(role="primary",candidate="kt_r",control="strongest_simple",
             label="KT-R minus the strongest development-selected simple learned control (M-I or M-C fitted on the same labels), per accepted primary contrast; controls changed from hand-authored rules to fitted models on 2 Oct 2026 (amendment proposed, awaiting confirmation)"),
        dict(role="mechanism_mean_collapse",candidate="kt_r",control="kt_r_mean",
             label="KT-R minus its mean-collapse twin (the same 49x42 readout on the weighted-mean hypothesis): does keeping the competing hypotheses matter?"),
        dict(role="mechanism_hard_collapse",candidate="kt_r",control="kt_r_hard",
             label="KT-R minus its hard-collapse twin (the same readout on the single most-supported hypothesis)"),
        dict(role="matched_control",candidate="kt_r",control="kt_set",
             label="KT-R minus KT-SET (matched set-pooling nonlinear control, primary contrast; isolates where pooling happens, not whether hypotheses are kept)"),
        dict(role="matched_information",candidate="kt_r",control="flat",
             label="KT-R minus the full-information flat readout (information- and size-matched; no hypothesis set)"),
        dict(role="secondary_representation",candidate="kt_set",control="flat",
             label="KT-SET minus full-information flat (representation diagnostic; not the primary contrast)"),
        dict(role="descriptive_floor",candidate="kt_r",control="strongest_rule",
             label="KT-R minus the stronger hand-authored rule (the 23 Sep M-I/M-C rules): descriptive floor only; the two rules decide identically at every grid threshold")),
    "rules_contrast":(  # the 23-24 Sep behaviour, kept so earlier reports can be reproduced
        dict(role="primary",candidate="kt_r",control="strongest_rule",
             label="KT-R minus the strongest development-selected simple learner (M-I or M-C), per accepted primary contrast (hand-authored rules; 23 Sep specification)"),
        dict(role="matched_control",candidate="kt_r",control="kt_set",
             label="KT-R minus KT-SET (matched set-pooling nonlinear control, primary contrast)"),
        dict(role="secondary_representation",candidate="kt_set",control="flat",
             label="KT-SET minus full-information flat (representation diagnostic; not the primary contrast)")),
    "representation":(
        dict(role="primary",candidate="kt_set",control="flat",
             label="KT-SET minus full-information flat (use as primary only with a recorded decision replacing primary contrast)"),),
}


def _proposes_on_test(fold):
    """Whether the arm proposed at least once on its held-out person; None when the fold carries no outcomes."""
    metrics=fold.get("reporting_metrics") or {}
    return _proposes(metrics) if "outcomes" in metrics else None


def _point_status(fold):
    if fold["threshold_selection"] not in ("development","risk_control"):return "fixed"
    return "feasible" if fold["development_budget_feasible"] else "infeasible"


def _development_success(fold):
    for point in fold.get("development_curve") or []:
        if point["threshold"]==fold["reporting_threshold"]:return point["success"]
    return None


def _exclusion_reason(fold):
    reason=fold.get("development_selection_reason","")
    if reason.startswith("silent"):return f"{fold['arm']} reporting point was silent: {reason}"
    return f"{fold['arm']} reporting point missed the development budget"


def contrasts_from_folds(folds,seeds,spec="primary_contrast",min_people_for_interval=MIN_PEOPLE_FOR_INTERVAL):
    """Per-person paired contrasts named by a recorded specification.

    * Success difference and unwanted-proposal burden difference (ticks and onsets) are reported together.
    * A person whose development-selected reporting point missed the declared budget, or was silent on the
      development data, for either arm is excluded from that contrast and listed with the reason; it is never
      averaged in. Nothing is excluded on TEST outcomes; with a fixed threshold no such exclusion exists, so every
      reporting point carries ``proposes_on_test`` and the contrast lists ``silent_arms_on_test`` (5 Oct 2026): a
      difference against an arm that issued nothing is that arm's silence, not an advantage.
    * A "strongest_*" control is chosen per person from development success only; with a fixed threshold it
      cannot be chosen, and that contrast is then reported as not selectable. Ties go to the first (simpler) arm.
    * Fixed-threshold contrasts are marked descriptive: no operating point was tuned.
    * Any average covers only the included people. It is reported with the feasibility of every arm's
      reporting point per person, and is never an estimate for everyone. A 95% paired t interval appears
      only from ``min_people_for_interval`` people upward; below that the field states why it is absent.
    """
    def fold_for(arm,person,seed):
        for f in folds:
            if f["arm"]==arm and f["held_out_person"]==person and (arm in FLOOR_ARMS or f["seed"]==seed):return f
        return None
    def value(f,person,key,field):
        return f["reporting_metrics"][key].get(person,{}).get(field)
    people=sorted({f["held_out_person"] for f in folds})
    contrasts=[]
    for seed in seeds:
        for entry in CONTRAST_SPECS[spec]:
            per_person={};burden={};onset_burden={};controls={};excluded={};points={};modes=set();not_selectable=0
            pool=STRONGEST.get(entry["control"])
            arms_involved=[entry["candidate"]]+(list(pool) if pool else [entry["control"]])
            feasibility={a:{"feasible":[],"budget_infeasible":[],"fixed_threshold":[]} for a in arms_involved}
            label_of={"feasible":"feasible","infeasible":"budget_infeasible","fixed":"fixed_threshold"}
            with_candidate=0
            for person in people:
                for a in arms_involved:
                    f=fold_for(a,person,seed)
                    if f is not None:feasibility[a][label_of[_point_status(f)]].append(person)
                cand=fold_for(entry["candidate"],person,seed)
                if cand is None:continue
                with_candidate+=1
                if pool:
                    options=[f for f in (fold_for(a,person,seed) for a in pool) if f is not None]
                    usable=[f for f in options if _point_status(f)=="feasible" and _development_success(f) is not None]
                    if not usable or any(_point_status(f)=="fixed" for f in options):
                        excluded[person]=f"{entry['control']} control cannot be development-selected (fixed threshold, missing arm, silent or budget-infeasible development points)"
                        not_selectable+=1;continue
                    ctrl=max(usable,key=lambda f:(_development_success(f),-pool.index(f["arm"])))  # tie: the simpler (first) arm
                else:
                    ctrl=fold_for(entry["control"],person,seed)
                    if ctrl is None:continue
                infeasible=[f for f in (cand,ctrl) if _point_status(f)=="infeasible"]
                if infeasible:
                    excluded[person]=_exclusion_reason(infeasible[0]);continue
                a,b=value(cand,person,"success","success_fraction"),value(ctrl,person,"success","success_fraction")
                per_person[person]=None if a is None or b is None else a-b
                ua,ub=value(cand,person,"unwanted","proposals_per_hand_minute"),value(ctrl,person,"unwanted","proposals_per_hand_minute")
                burden[person]=None if ua is None or ub is None else ua-ub
                oa,ob=value(cand,person,"unwanted","onsets_per_hand_minute"),value(ctrl,person,"unwanted","onsets_per_hand_minute")
                onset_burden[person]=None if oa is None or ob is None else oa-ob
                controls[person]=ctrl["arm"]
                points[person]={f["arm"]:dict(selection=f["threshold_selection"],threshold=f["reporting_threshold"],
                                              budget_feasible=f["development_budget_feasible"],
                                              proposes_on_test=_proposes_on_test(f)) for f in (cand,ctrl)}
                modes|={_point_status(cand),_point_status(ctrl)}
            known=[d for d in per_person.values() if d is not None]
            if pool and not per_person and not_selectable:
                status="not_selectable_without_development_split"
            elif "fixed" in modes:status="descriptive_fixed_threshold"
            elif excluded:status="development_selected_some_people_excluded"
            elif per_person:status="development_selected_and_feasible"
            else:status="no_comparable_people"
            contrasts.append(dict(role=entry["role"],label=entry["label"],seed=seed,candidate=entry["candidate"],
                control=entry["control"],control_by_person=controls,per_person=per_person,excluded_people=excluded,
                people_with_candidate=with_candidate,n_people=len(known),feasibility=feasibility,
                mean_difference_among_included_people=(sum(known)/len(known) if known else None),
                min_difference_among_included_people=(min(known) if known else None),
                max_difference_among_included_people=(max(known) if known else None),
                interval_95_among_included_people=paired_interval(known,min_people_for_interval),
                population_note=(f"average over the {len(known)} of {with_candidate} people whose points were usable for both arms; "
                                 "excluded people and every arm's feasibility are listed; not an estimate for all people"),
                burden_difference_proposals_per_hand_minute=burden,
                burden_difference_onsets_per_hand_minute=onset_burden,reporting_points=points,status=status,
                silent_arms_on_test={person:[arm for arm,point in arms.items() if point["proposes_on_test"] is False]
                                     for person,arms in points.items() if any(point["proposes_on_test"] is False for point in arms.values())},
                scope="participant-level paired differences of success fraction; burden differences (ticks and onsets) beside it; interval only from the declared minimum number of people"))
    return contrasts
