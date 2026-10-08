"""Sequential real-runtime validation driver; imports no model implementation.

Waits for the already-started baseline video_01, then invokes real production
CLIs. Does not change architecture/config, kill other workloads, or retry OOM
with reduced settings. State/logs are confined to final_runtime_validation.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/final_runtime_validation'
PYTHON = '/home/work/ai-graphics_33/miniconda3/envs/Hrlrvo/bin/python'
ENV = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'PYTHONPATH': str(ROOT / 'main/src'),
       'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'}
STATE = {'started_at': datetime.now(timezone.utc).isoformat(), 'jobs': []}
STATE_FILE = OUT / 'execution_state.json'


def persist():
    STATE_FILE.write_text(json.dumps(STATE, indent=2) + '\n')


def gpu_status():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free',
                                    '--format=csv'], text=True)


def wait_idle():
    while True:
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                             '--format=csv,noheader'], text=True).strip()
        if not processes:
            return
        STATE['waiting_for_gpu'] = processes
        persist()
        time.sleep(30)


def run(name, command, folder):
    wait_idle()
    STATE.pop('waiting_for_gpu', None)
    folder.mkdir(parents=True, exist_ok=True)
    record = {'name': name, 'command': command, 'started_at': datetime.now(timezone.utc).isoformat(),
              'gpu_before': gpu_status(), 'status': 'RUNNING', 'log': str(folder / (name + '.launcher.log'))}
    STATE['jobs'].append(record)
    persist()
    started = time.monotonic()
    with Path(record['log']).open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
        record['pid'] = process.pid
        persist()
        with (folder / (name + '.gpu_trace.jsonl')).open('w') as trace:
            while process.poll() is None:
                trace.write(json.dumps({'time': datetime.now(timezone.utc).isoformat(), 'gpu': gpu_status()}) + '\n')
                trace.flush()
                time.sleep(15)
        code = process.wait()
    record.update({'status': 'PASS' if code == 0 else 'FAIL', 'exit_code': code,
                   'elapsed_seconds': time.monotonic() - started, 'gpu_after': gpu_status()})
    persist()
    return code == 0


def main():
    global STATE_FILE
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-pid', type=int, required=True)
    parser.add_argument('--training-only', action='store_true', help='Retry a corrected runtime observer, preserving failed artifacts')
    args = parser.parse_args()
    if args.training_only:
        STATE_FILE = OUT / 'execution_retry_state.json'
    persist()
    while True:
        try:
            os.kill(args.wait_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(15)
    if not args.training_only:
        baseline = OUT / 'pretrain_inference/video_01/inference_manifest.json'
        STATE['jobs'].append({'name': 'pretrain_video_01', 'status': 'PASS' if baseline.is_file() else 'FAIL',
                              'manifest': str(baseline)})
    persist()
    train = [PYTHON, '-u', '-m', 'ltx_outpaint.train', str(ROOT / 'main/configs/train_4k.yaml')]
    smoke = OUT / 'training_smoke'
    if args.training_only:
        smoke = smoke / 'validated_run'
    passed = run('training_initial', train + ['--output-dir', str(smoke), '--stop-after-step', '5',
                                             '--disable-progress-bars'], smoke)
    checkpoint = smoke / 'checkpoints/checkpoint-000005'
    if passed:
        run('training_resume', train + ['--resume', str(checkpoint), '--stop-after-step', '6',
                                       '--disable-progress-bars'], smoke)
    else:
        STATE['jobs'].append({'name': 'training_resume', 'status': 'SKIPPED_DEPENDENCY'})
        persist()
    # User narrowed validation to the first two videos; video_03 is excluded.
    videos = [] if args.training_only else [(2, '0f2313f5-5a61-4ccc-811c-f3bbf1d41151')]
    for index, name in videos:
        folder = OUT / f'pretrain_inference/video_{index:02d}'
        command = [PYTHON, '-u', '-m', 'ltx_outpaint.inference', '--config', str(ROOT / 'main/configs/inference_4k.yaml'),
                   '--anchor-video', str(ROOT / f'input/anchor/{name}.mp4'), '--output', str(folder / 'generated.mp4'),
                   '--initialized-adapters', '--seed', '42', '--steps', '30']
        run(f'pretrain_video_{index:02d}', command, folder)
    if passed:
        folder = OUT / 'checkpoint_load_inference'
        command = [PYTHON, '-u', '-m', 'ltx_outpaint.inference', '--config', str(ROOT / 'main/configs/inference_4k.yaml'),
                   '--anchor-video', str(ROOT / 'input/anchor/ef334007-aa0b-432d-a7e8-76a3eeeec252.mp4'),
                   '--output', str(folder / 'generated.mp4'), '--checkpoint', str(checkpoint), '--num-frames', '65',
                   '--seed', '42', '--steps', '30']
        run('checkpoint_inference', command, folder)
    else:
        STATE['jobs'].append({'name': 'checkpoint_inference', 'status': 'SKIPPED_DEPENDENCY'})
    STATE['finished_at'] = datetime.now(timezone.utc).isoformat()
    persist()


if __name__ == '__main__':
    main()
