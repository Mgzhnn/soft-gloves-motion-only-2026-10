"""Full-check perception regressions; synthetic offline software, participant N=0."""
import json
import numpy as np
import pytest
import cv2
from test_object_providers import make_recording, two_box_scene, CAL, SETTINGS
from test_compare_object_providers import write_run
import compare_object_providers as cmp
import segment_objects
from run_perception import build_object_provider
from sg_offline.objects import MaskObjects, PlaneClusterObjects
from sg_offline.perception import AnatomicalTracker, HandDetection
from sg_offline.ledger import FUNCTIONAL_POINTS

def test_comparison_fixed_latency_is_a_timing_setting(tmp_path):
    a=write_run(tmp_path,'a','colour',{0:{'cup':(0.,0.,.6)}},{})
    b=write_run(tmp_path,'b','masks',{0:{'cup':(0.,0.,.6)}},{})
    for path, latency in [(a,.1),(b,.2)]:
        p=path/'summary.json'; d=json.loads(p.read_text()); d['fixed_latency_s']=latency; d['manifest_sha256']='same'; p.write_text(json.dumps(d))
    report=cmp.compare([a,b])
    assert report['identical_timing_settings'] is False, report

def test_mask_build_requires_identity_provenance(tmp_path):
    rec=make_recording(tmp_path,n=1)
    seg=tmp_path/'foreign_masks'; seg.mkdir()
    (seg/'objects.json').write_text(json.dumps({'schema':'sg-object-masks/1','objects':{'1':'foreign_cup'}}))
    with pytest.raises(ValueError):
        build_object_provider('masks',rec.calibration,masks_dir=seg,recording=rec)

def test_segmenter_duplicate_indices_cannot_declare_complete_set(tmp_path):
    rec=make_recording(tmp_path,n=3)
    class DuplicateSegmenter:
        def segment_chunk(self,*args):
            for idx in [0,0,2]:
                yield idx,{1:np.ones((48,64),dtype=bool)}
    with pytest.raises((ValueError,RuntimeError)):
        prov=segment_objects.run_segmentation(rec,tmp_path/'seg',{'objects':{'cup':{'points':[[10,10]],'labels':[1]}}},DuplicateSegmenter(),chunk=3)
        print('declared',prov['frames_written'],'actual',len(list((tmp_path/'seg'/'masks').glob('*.png'))))

def test_unknown_mask_label_is_not_silently_absent(tmp_path):
    depth,box,second,arm=two_box_scene()
    seg=tmp_path/'seg'; (seg/'masks').mkdir(parents=True)
    (seg/'objects.json').write_text(json.dumps({'schema':'sg-object-masks/1','objects':{'1':'cup'},'first_source_index':0,'frames_written':1}))
    label=np.zeros(depth.shape,np.uint8); label[box]=2
    cv2.imwrite(str(seg/'masks'/'00000.png'),label)
    with pytest.raises(ValueError):
        result=MaskObjects(CAL,seg,SETTINGS).observe(depth,0)
        print('wrong_label_result',result['cup'].reason)

def test_adapter_rejects_missing_host_arrival_instead_of_fabricating_time(tmp_path):
    from test_image_sequence_adapter import make_export, calibration
    import image_sequence_adapter as adapter
    source=tmp_path/'export'; stem=make_export(source,n=1)
    (source/f'{stem}_camera.txt').write_text('host_time\tframe_index\nnan\t0\n')
    with pytest.raises(ValueError):
        result=adapter.convert(source,tmp_path/'out','p','s',calibration=calibration(),calibration_source='synthetic',registered_to_rgb=True,min_free_gib=0)
        frame=next(iter(adapter.RGBDRecording(tmp_path/'out'/'recording.json')))
        print('invalid_host_input=nan output_rgb_delay_s=',frame.rgb_stamp.available_s-frame.rgb_stamp.acquired_s,'frames_written=',result['frames_written'])
