from __future__ import annotations
import json,math
from pathlib import Path
from src.log_reader import IncrementalJsonlReader
from src.metrics import block_figure,block_records,canonicalize_record,health_summary,scalar_frame,status_from,sustained_zero_gradient_groups

ROOT=Path(__file__).resolve().parents[1]

def test_mock_jsonl_parses_and_has_48_blocks():
    snap=IncrementalJsonlReader(ROOT/'mock_data/train_metrics.jsonl').refresh()
    assert len(snap.records)==160 and snap.malformed_lines==0
    latest=canonicalize_record(snap.records[-1]); assert latest['optimizer_step']==160
    assert len(latest['local_native_ratio_by_block'])==48
    assert len(block_records(snap.records))==160
    assert len(block_figure(snap.records[-1],('Native','local_native_ratio_by_block'),('Adaptive','local_adaptive_ratio_by_block'),'x').data)==2


def test_aliases_and_missing_metrics_are_flexible():
    record={'global_step':7,'train_loss':.5,'lr':1e-4,'op_lora_grad_norm':.2}
    c=canonicalize_record(record); assert c['optimizer_step']==7 and c['loss']==.5 and c['grad_norm_op_lora']==.2
    assert c['grad_norm_r4'] is None
    frame=scalar_frame([record]); assert frame.loc[0,'optimizer_step']==7


def test_nan_inf_and_frozen_health_are_critical():
    latest=canonicalize_record({'nan_gradient_count':1,'inf_gradient_count':2,'frozen_parameter_grad_violation_count':3,'zero_gradient_trainable_groups':['R4']})
    health=health_summary(latest,[]); assert health['level']=='critical'; assert len(health['critical'])==3
    assert health['warnings']==['Zero-gradient groups: R4']


def test_json_nan_does_not_crash_scalar_handling(tmp_path):
    p=tmp_path/'x.jsonl'; p.write_text('{"optimizer_step":1,"loss":NaN,"nan_gradient_count":1}\n')
    snap=IncrementalJsonlReader(p).refresh(); frame=scalar_frame(snap.records)
    assert len(frame)==1 and math.isnan(frame.loc[0,'loss'])


def test_status_thresholds():
    latest={'timestamp':None}
    assert status_from(latest,100,now=105)[0]=='RUNNING'
    assert status_from(latest,100,now=130)[0]=='STALE'
    assert status_from(latest,100,now=200)[0]=='STOPPED / NO UPDATE'
    assert status_from({},None,now=200)[0]=='ERROR'


def test_sustained_zero_gradient_warning():
    frame=scalar_frame([{'optimizer_step':i,'grad_norm_r4':0.0,'grad_norm_c2':0.1} for i in range(12)])
    assert sustained_zero_gradient_groups(frame)==['R4']
