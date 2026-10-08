from __future__ import annotations
import json,os
from pathlib import Path
from src.log_reader import IncrementalJsonlReader,ReaderRegistry,discover_runs,read_last_lines,read_text_artifact


def line(**values): return (json.dumps(values)+'\n').encode()

def test_incremental_tail_and_incomplete_final_line(tmp_path):
    path=tmp_path/'train_metrics.jsonl'; path.write_bytes(line(step=1)+line(step=2)+b'{"step":3')
    reader=IncrementalJsonlReader(path); first=reader.refresh()
    assert [x['step'] for x in first.records]==[1,2]
    assert first.pending_bytes>0; offset=first.byte_offset
    with path.open('ab') as f: f.write(b'}\n')
    second=reader.refresh()
    assert [x['step'] for x in second.records]==[1,2,3]
    assert second.byte_offset>offset and second.pending_bytes==0


def test_malformed_line_is_skipped_without_crash(tmp_path):
    path=tmp_path/'train_metrics.jsonl'; path.write_bytes(line(step=1)+b'not json\n'+line(step=2))
    snap=IncrementalJsonlReader(path).refresh()
    assert [x['step'] for x in snap.records]==[1,2]
    assert snap.malformed_lines==1 and snap.last_error is None


def test_replacement_and_truncation_reload_safely(tmp_path):
    path=tmp_path/'train_metrics.jsonl'; path.write_bytes(line(step=1)+line(step=2))
    reader=IncrementalJsonlReader(path); reader.refresh()
    replacement=tmp_path/'replacement'; replacement.write_bytes(line(step=10)); os.replace(replacement,path)
    snap=reader.refresh(); assert [x['step'] for x in snap.records]==[10]; assert snap.reset_count==1
    path.write_bytes(line(step=20))
    snap=reader.refresh(); assert [x['step'] for x in snap.records]==[20]; assert snap.reset_count==2


def test_run_switch_keeps_independent_offsets(tmp_path):
    a=tmp_path/'a.jsonl'; b=tmp_path/'b.jsonl'; a.write_bytes(line(step=1)); b.write_bytes(line(step=9))
    registry=ReaderRegistry(); assert registry.get(a) is registry.get(a)
    assert registry.get(a) is not registry.get(b)
    assert registry.get(a).refresh().records[-1]['step']==1
    assert registry.get(b).refresh().records[-1]['step']==9
    with a.open('ab') as f: f.write(line(step=2))
    assert [x['step'] for x in registry.get(a).refresh().records]==[1,2]


def test_discovery_newest_first_and_direct_priority(tmp_path):
    for name,step,mtime in [('old',1,100),('new',2,200)]:
        d=tmp_path/name; d.mkdir(); p=d/'train_metrics.jsonl'; p.write_bytes(line(optimizer_step=step)); os.utime(p,(mtime,mtime))
    found=discover_runs(tmp_path); assert [x['run_name'] for x in found]==['new','old']; assert found[0]['latest_step']==2
    direct=discover_runs(tmp_path,direct_log=tmp_path/'old/train_metrics.jsonl'); assert len(direct)==1 and direct[0]['run_name']=='old'


def test_console_and_artifact_missing_are_safe(tmp_path):
    assert read_last_lines(tmp_path/'missing.log')=='console log unavailable'
    assert read_text_artifact(tmp_path/'missing.json')=='unavailable'
    p=tmp_path/'console.log'; p.write_text('a\nb\nc\n'); assert read_last_lines(p,2)=='b\nc'
