#!/usr/bin/env python3
"""Wait for the drained original run and continue ONLY video 01 outpainting.

Own state/logs are separate from production's historical batch journal. No
signal, deletion, config change, or upscale invocation exists in this script.
"""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def process_matches(pid: int, start_ticks: str) -> bool:
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[19] == start_ticks and fields[0] != 'Z'


def locks_free(root: Path) -> bool:
    for relative in ['state/batch_runner.lock', 'state/locks/outpaint.lock',
                     'state/locks/video_02.lock', 'state/locks/video_01.lock']:
        path = root / relative
        if not path.exists():
            continue
        with path.open('rb') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            fcntl.flock(handle, fcntl.LOCK_UN)
    return True


def validate_completed_02(root: Path, config: dict, *, hash_output: bool = True) -> dict:
    stage = root / config['artifacts']['outpaint']
    receipt = json.loads((stage / 'metadata/stage_complete.json').read_text())
    identity = {key: config[key] for key in ['input', 'seed', 'frames', 'preprocess', 'outpainting']}
    if 'validation' in config:
        identity['validation'] = config['validation']
    output = stage / 'final/02_Outpaint_4K.mp4'
    require(config['input']['video_id'] == '02', 'Wrong predecessor video')
    require(receipt.get('status') == 'PASS', '02 has no PASS receipt')
    require(receipt.get('request') == identity, '02 config identity mismatch')
    require(Path(receipt['output']).resolve() == output.resolve(), '02 output mismatch')
    require(receipt.get('frames') == 1952 and receipt.get('output_fps') == 16, '02 frame/fps mismatch')
    require(output.is_file() and output.stat().st_size > 0, '02 output missing or empty')
    if hash_output:
        require(sha256(output) == receipt['output_sha256'], '02 output hash mismatch')
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    root, state = Path(plan['production_root']), args.plan.parent
    configs = {}
    for video in ['02', '01']:
        path = root / f'configs/certification/{video}.json'
        require(sha256(path) == plan['config_sha256'][video], f'{video} config changed')
        configs[video] = json.loads(path.read_text())
        require(configs[video]['input']['video_id'] == video, 'Input video mismatch')
    python = Path(plan['outpaint_python'])
    require(python.is_file() and os.access(python, os.X_OK), 'Outpaint Python unavailable')
    require((root / 'run_cert.sh').is_file(), 'Stage launcher unavailable')
    command = [str(root / 'run_cert.sh'), '--config', str(root / 'configs/certification/01.json'),
               '--stage', 'outpaint', '--resume']
    if args.preflight:
        print(json.dumps({'preflight': 'PASS', 'command': command, 'signals': False,
                          'upscale_invocation': False}))
        return 0
    lock = (state / 'continuation.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def event(status: str, **details) -> None:
        record = {'time_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                  'status': status, 'pid': os.getpid(), **details}
        temporary = state / '.continuation_state.tmp'
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(state / 'continuation_state.json')
        print(json.dumps(record), flush=True)

    try:
        event('WAITING_FOR_DRAINED_02', watched_processes=plan['wait_processes'])
        while any(process_matches(x['pid'], x['start_ticks']) for x in plan['wait_processes']) or not locks_free(root):
            time.sleep(15)
        # A graceful parent exit alone is not proof of successful outpainting.
        for video in ['02', '01']:
            require(sha256(root / f'configs/certification/{video}.json') == plan['config_sha256'][video], f'{video} config changed while waiting')
        receipt = validate_completed_02(root, configs['02'])
        batch_log = (root / 'logs/batch_pipeline.log').read_text()
        require(plan['required_drain_log'] in batch_log, 'drain was never recorded')
        existing = root / configs['01']['artifacts']['outpaint'] / 'metadata/stage_complete.json'
        if existing.exists():
            raise RuntimeError('01 already has a receipt; refuse an unreviewed duplicate run')
        env = dict(os.environ, CERT_OUTPAINT_PYTHON=str(python),
                   CERT_FLASH_PYTHON=plan['flash_python'], PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        event('STARTING_01_OUTPAINT_ONLY', previous_output=receipt['output'], command=command)
        # A durable file, not a pipe owned by this watcher: 01 keeps writing if
        # this supervisor disappears. Its process session is also independent.
        child_log = state / '01_outpaint.log'
        with child_log.open('a', buffering=1) as handle:
            child = subprocess.Popen(command, cwd=root, env=env, start_new_session=True,
                                     stdout=handle, stderr=subprocess.STDOUT)
            event('01_OUTPAINT_RUNNING', child_pid=child.pid, command=command,
                  child_log=str(child_log))
            code = child.wait()
        if code:
            raise RuntimeError(f'01 outpaint exited {code}; no automatic retry or upscale')
        final_receipt = json.loads(existing.read_text())
        require(final_receipt.get('status') == 'PASS', '01 has no PASS receipt')
        final_path = Path(final_receipt['output'])
        require(final_path.is_file() and sha256(final_path) == final_receipt['output_sha256'], '01 final hash mismatch')
        event('01_OUTPAINT_COMPLETE', output=str(final_path), upscale_scheduled=False)
        return 0
    except Exception as error:
        event('BLOCKED_NO_DESTRUCTIVE_RECOVERY', error=repr(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
