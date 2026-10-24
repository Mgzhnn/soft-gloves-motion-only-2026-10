"""Regression tests for failure cases that contract-only tests previously missed. N=0 fixtures."""
from dataclasses import replace
import json
import numpy as np
import pytest
from sg_offline.fixtures import scenarios
from sg_offline.inference import LinearJoint, FlatJoint, HypothesisPoolMLP, flat_features, label_indices
from sg_offline.training import bundles_for, label_ticks, mean_log_loss
from sg_offline.evaluation import Opportunity, evaluate_opportunities, timeline_accounting
from sg_offline.replay import Replay, SimulationSupervisor, Config
from sg_offline.stats import unwanted_exposure
from sg_offline.ledger import FUNCTIONAL_POINTS, LandmarkLabel, read_landmark_ledger, write_landmark_ledger, to_opportunities, _entry
from sg_offline.perception import HandDetection, AnatomicalTracker, packet_from_detections, ColourObjects, match_boxes
from test_rgbd import frame


def test_partial_label_gradient_marginalises_unknown_hand():
    b=list(bundles_for(scenarios()))[2][1];model=LinearJoint()
    model.W=np.random.default_rng(3).normal(0,.01,model.W.shape)
    allowed=tuple(i*7+1 for i in range(7))
    loss,grad=model.loss_gradient(b,allowed)
    assert loss==pytest.approx(-np.log(model.predict(b)[list(allowed)].sum()))
    eps=1e-6;model.W[0,0]+=eps;plus=model.loss_gradient(b,allowed)[0]
    model.W[0,0]-=2*eps;minus=model.loss_gradient(b,allowed)[0]
    assert grad[0,0]==pytest.approx((plus-minus)/(2*eps),abs=1e-7)


def test_left_exclusion_preserves_right_grasp_and_respects_contact_window():
    packets=scenarios();p=packets[0]
    op=Opportunity('r',p.participant_id,p.session_id,.5,1.4,(None,('G',)),'fixture',((.5,1.4),(.7,1.1)))
    examples,c=label_ticks(packets,[op])
    assert examples and all(.7<=e.decision_s<1.1 for e in examples)
    assert all(set(label_indices(e.label))=={i*7+1 for i in range(7)} for e in examples)
    assert c['partial_reference']==len(examples)


def test_per_hand_scope_preserves_grasp_when_other_action_unimplemented():
    e=_entry(dict(opportunity_id='x',kind='request',start_s=0,end_s=1,
        acceptable={'left':['R'],'right':['G']},provenance='fixture',annotator='a',prefix_only=False))
    inside,out=to_opportunities(dict(clock={'kind':'trace'},participant_id='p',session_id='s',entries=[e]))
    assert inside[0].acceptable==(None,('G',)) and out[0].acceptable==(('R',),('G',))


def test_marginal_request_is_not_suppressed_by_joint_mode():
    packet,b=list(bundles_for(scenarios()))[2]
    scores=np.zeros((7,7));scores[0,0]=.36;scores[1,0]=.32;scores[1,1]=.32
    out=SimulationSupervisor(Config()).admit(packet,scores,b)
    assert out[0]['raw_outcome']=='G' and out[0]['outcome_score']==pytest.approx(.64)
    assert out[0]['status']=='proposed_simulation_only'


def test_flat_contains_all_candidates_and_is_order_invariant():
    bundle=list(bundles_for(scenarios()))[7][1]
    assert len(bundle.hypotheses)==19
    order=list(reversed(range(19)))
    perm=replace(bundle,hypotheses=tuple(bundle.hypotheses[i] for i in order),features=bundle.features[order],weights=bundle.weights[order])
    assert np.array_equal(flat_features(bundle),flat_features(perm))
    # the flat input encodes one row per object (sum and difference products, see
    # inference.flat_features); every hypothesis weight sits in a row or in the both-hands-on-nothing slot.
    from sg_offline.inference import flat_object_rows
    r1,r2=flat_object_rows(bundle);none_none=sum(w for h,w in zip(bundle.hypotheses,bundle.weights) if h.object_ids==(None,None))
    assert (r1+r2)[10:].sum()+none_none==pytest.approx(1.)


def test_unwanted_burden_is_invariant_to_splitting_intervals_and_counts_both_hands():
    rows=Replay().run(scenarios());p=rows[0]['inputs']
    for row in rows:
        for out in row['outputs']:out.update(status='proposed_simulation_only',raw_outcome='G')
    def op(name,start,end):return Opportunity(name,p['participant_id'],p['session_id'],start,end,(('NQ',),('NQ',)),'fixture')
    whole=[op('a',.5,1.4)];split=[op('b',.5,.8),op('c',.8,1.4)]
    a=unwanted_exposure(evaluate_opportunities(rows,whole),whole)[p['participant_id']]
    b=unwanted_exposure(evaluate_opportunities(rows,split),split)[p['participant_id']]
    assert a['unwanted_proposals']==b['unwanted_proposals']==18
    assert a['proposals_per_hand_minute']==pytest.approx(b['proposals_per_hand_minute'])
    assert a['exposure_hand_s']==pytest.approx(1.8)


def test_outside_reference_is_counted_without_calling_it_no_request():
    rows=Replay().run(scenarios());counts=timeline_accounting(rows,[])
    assert counts['unlabelled_hand_ticks']==18 and 'no_request_hand_ticks' not in counts


def detection(side='left',box=(0,0,6,4)):
    return HandDetection(box,side,.9,{p:(2+i,2) for i,p in enumerate(FUNCTIONAL_POINTS)},'fixture')


def test_identity_deduplicates_and_does_not_use_image_side():
    tracker=AnatomicalTracker()
    d=detection('right');out=tracker.assign([d,replace(d,confidence=.8)],1.)
    assert out['left'] is None and out['right']==d
    ambiguous=AnatomicalTracker().assign([d,replace(d,side='left')],1.)
    assert all(v is None for v in ambiguous.values())


def test_identity_does_not_relabel_a_mirrored_pose_after_crossing():
    tr=AnatomicalTracker();left=detection('left',(0,0,20,20));right=detection('right',(80,0,100,20))
    tr.assign([left,right],1.)
    out=tr.assign([replace(left,box=right.box),replace(right,box=left.box)],1.1)
    assert out=={'left':None,'right':None}


def test_bridge_requires_depth_at_all_three_points_not_background():
    f=frame();d=detection();depth=f.depth.copy();depth[2,3]=0;f=replace(f,depth=depth)
    packet,diag=packet_from_detections(f,{'left':d}, {'cup':(5,3)},participant_id='p',session_id='s',available_s=1.1)
    assert packet.hands[0].xyz is None and packet.objects[0].xyz is not None
    assert diag['left']['points']['index_mcp']=='invalid_depth'
    assert packet.hands[0].stamp.acquired_s==1 and packet.hands[0].stamp.available_s==1.1


def test_two_colour_targets_stay_distinct_from_black_glove():
    image=np.zeros((30,60,3),np.uint8);image[3:18,4:19]=[0,200,0];image[5:20,35:50]=[220,0,0]
    pix=ColourObjects().pixels(image)
    assert pix['green_cup'][0]<30 and pix['red_marker'][0]>30


def test_box_matching_does_not_double_count_one_hand():
    assert len(match_boxes([(0,0,20,20),(1,1,21,21)],[(0,0,20,20),(80,0,100,20)]))==1


def test_box_label_and_annotator_isolation(tmp_path):
    box=LandmarkLabel('f','p','s','left','hand_box','visible',None,None,'a','unmarked',box=(0,0,6,4))
    path=tmp_path/'a.jsonl';write_landmark_ledger(path,[box]);assert read_landmark_ledger(path)==(box,)
    with pytest.raises(ValueError,match='separate'):write_landmark_ledger(path,[box,replace(box,annotator='b')])
    path.write_text(json.dumps(box.to_row())+'\n'+json.dumps(replace(box,annotator='b').to_row())+'\n')
    with pytest.raises(ValueError,match='separate'):read_landmark_ledger(path)


def test_request_import_requires_complete_timeline_and_preserves_other_hand():
    from import_requests import convert
    rows=[dict(start_s='0',end_s='1',left='EXCLUDED',right='G',note='left holding')]
    data=convert(rows,'p','s','a','clock',1.)
    assert data['opportunities'][0]['acceptable']=={'left':None,'right':['G']}
    with pytest.raises(ValueError,match='duration'):convert(rows,'p','s','a','clock',2.)


def test_comparison_keeps_people_separate_and_reduces_loss():
    from sg_offline.experiment import compare
    sessions=[]
    for person in ('p1','p2'):
        packets=[replace(p,participant_id=person,session_id='s') for p in scenarios()]
        ops=[Opportunity('a',person,'s',.5,.9,(('G',),('NQ',)),'synthetic_fixture'),
             Opportunity('b',person,'s',.9,1.4,(('NQ',),('NQ',)),'synthetic_fixture')]
        sessions.append(dict(packets=packets,opportunities=ops))
    result=compare(sessions,arms=('flat','kt_set','kt_r'),steps=30,thresholds=(.5,.6),synthetic_check=True)
    assert len(result['folds'])==6 and all(f['fitting_improved'] for f in result['folds'])
    assert all(f['held_out_person'] not in {p for p,s in f['train_sessions']} for f in result['folds'])
    assert all(f['reporting_threshold']==.6 for f in result['folds'])
    # contrasts follow; without M-I/M-C arms the primary one is not
    # selectable, so check the matched KT-R minus KT-SET contrast (artificial identifiers, N=0 real people).
    matched=[c for c in result['paired_contrasts'] if c['role']=='matched_control'][0]
    assert matched['n_people']==2 and matched['status']=='descriptive_fixed_threshold'


def test_perception_evaluator_matches_geometry_and_reports_wrong_identity():
    from evaluate_perception import evaluate
    labels=[LandmarkLabel('frame','p','s','left','hand_box','visible',None,None,'a','unmarked',box=(0,0,20,20)),
            LandmarkLabel('frame','p','s','right','hand_box','out_of_frame',None,None,'a','unmarked'),
            LandmarkLabel('frame','p','s','left','thumb_mcp','visible',5,5,'a','unmarked')]
    row=dict(participant_id='p',session_id='s',frame_id='frame',detections=[dict(box=(0,0,20,20),side='right',points={'thumb_mcp':(5,5)})])
    result=evaluate([row],labels)
    assert result['counts']['matched_hands']==1 and result['counts']['handedness_correct']==0
    assert result['point_error_px']['thumb_mcp']['median']==0
    row['session_id']='different-session'
    result=evaluate([row],labels)
    assert result['counts']['unlabelled_prediction_frames']==1


def test_export_requires_complete_hands_and_never_splits_test_person(tmp_path):
    from dataclasses import asdict
    from sg_offline.recordings import RECORDING_SCHEMA
    from prepare_hand_training import export
    f=frame();np.save(tmp_path/'rgb.npy',f.rgb);np.save(tmp_path/'depth.npy',f.depth)
    entries=[]
    for split,person in [('train','p1'),('val','p1'),('test','p2')]:
        frame_id='frame-'+split
        manifest=dict(schema=RECORDING_SCHEMA,participant_id=person,session_id=split,camera_model='synthetic',calibration=asdict(f.calibration),
            frames=[dict(frame_id=frame_id,rgb_path='rgb.npy',depth_path='depth.npy',rgb_stamp=asdict(f.rgb_stamp),depth_stamp=asdict(f.depth_stamp),registered_to_rgb=True,max_sync_skew_s=0)])
        path=tmp_path/(split+'.json');path.write_text(json.dumps(manifest))
        labels=[LandmarkLabel(frame_id,person,split,'left','hand_box','visible',None,None,'a','unmarked',box=(0,0,6,4)),
                LandmarkLabel(frame_id,person,split,'right','hand_box','out_of_frame',None,None,'a','unmarked')]
        labels += [LandmarkLabel(frame_id,person,split,'left',p,'visible',2+i,2,'a','unmarked') for i,p in enumerate(FUNCTIONAL_POINTS)]
        ledger=tmp_path/(split+'.jsonl');write_landmark_ledger(ledger,labels)
        entries.append(dict(split=split,manifest=str(path),ledger=str(ledger)))
    out=tmp_path/'export';report=export(dict(sessions=entries),out,'pose')
    assert report['frames']=={'train':1,'val':1,'test':1}
    assert json.loads((out/'dataset.yaml').read_text())['kpt_shape']==[3,3]
    text=next((out/'labels/train').glob('*.txt')).read_text().split()
    assert len(text)==14 and text[0]=='0'
    # Same test person under another session is still leakage.
    altered=json.loads((tmp_path/'test.json').read_text());altered['participant_id']='p1'
    (tmp_path/'test.json').write_text(json.dumps(altered))
    labels=[replace(l,participant_id='p1') for l in read_landmark_ledger(tmp_path/'test.jsonl')]
    write_landmark_ledger(tmp_path/'test.jsonl',labels)
    with pytest.raises(ValueError,match='leaked'):export(dict(sessions=entries),tmp_path/'bad','pose')


def test_asynchronous_or_different_calibration_objects_cannot_supply_geometry():
    packet=scenarios()[1]
    obj=packet.objects[0]
    for bad in (replace(obj,stamp=replace(obj.stamp,acquired_s=obj.stamp.acquired_s-.01)),replace(obj,calibration_id='different')):
        bundle=list(bundles_for([replace(packet,objects=(bad,))]))[0][1]
        assert all(h.object_ids==(None,None) for h in bundle.hypotheses)
        assert np.all(bundle.features[:,[0,15]]==0)


def test_fully_excluded_interval_is_retained_for_coverage():
    e=_entry(dict(opportunity_id='x',kind='request',start_s=.5,end_s=1.4,
        acceptable={'left':None,'right':None},provenance='fixture',annotator='a',prefix_only=False))
    inside,_=to_opportunities(dict(clock={'kind':'trace'},participant_id='synthetic-person',session_id='synthetic-session',entries=[e]))
    counts=timeline_accounting(Replay().run(scenarios()),inside)
    assert counts['unknown_or_excluded_hand_ticks']==18 and 'unlabelled_hand_ticks' not in counts


def test_request_import_rejects_unknown_duration():
    from import_requests import convert
    with pytest.raises(ValueError,match='finite'):convert([], 'p','s','a','clock',float('nan'))


def test_flat_does_not_receive_hidden_object_name_information():
    bundle=list(bundles_for(scenarios()))[7][1]
    renamed=replace(bundle,hypotheses=tuple(replace(h,object_ids=tuple(None if s is None else 'z_'+s[::-1] for s in h.object_ids)) for h in bundle.hypotheses))
    assert np.array_equal(flat_features(bundle),flat_features(renamed))
