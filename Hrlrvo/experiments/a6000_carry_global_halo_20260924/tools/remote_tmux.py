"""Run an uploaded task in a new, visible hrlrvo tmux window; never paste into a busy pane."""
from pathlib import Path
import argparse
import base64
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REMOTE = '/home/oem/hrlrvo/experiments/a6000_carry_global_halo_20260924'
HOST = 'oem@203.253.25.176'
OPTIONS = ['-i', str(Path.home()/'.ssh/codex_school_a6000_20260923_ed25519'),
           '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes',
           '-o', 'StrictHostKeyChecking=yes', '-o',
           'UserKnownHostsFile='+str(Path.home()/'.ssh/known_hosts_codex_school_a6000')]

def ssh(command):
    return subprocess.check_output(['ssh', '-p', '2222', *OPTIONS, HOST, command], text=True, timeout=30)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('script', type=Path)
    p.add_argument('--name', required=True)
    p.add_argument('--wait', action='store_true')
    p.add_argument('--timeout', type=int, default=900)
    a = p.parse_args()
    tag = a.name+'_'+str(time.time_ns())
    payload = base64.b64encode(a.script.read_bytes()).decode()
    log = f'{REMOTE}/admin/{tag}.log'
    script = f'{REMOTE}/admin/{tag}.sh'
    marker = 'REMOTE_TASK_DONE_'+tag
    command = (f'umask 077; mkdir -p {REMOTE}/admin; '
               f'printf %s {shlex.quote(payload)} | base64 -d > {script}; '
               f'bash {script} > >(tee {log}) 2>&1; rc=$?; '
               f'echo "$rc" > {REMOTE}/admin/{tag}.exit; '
               f'echo {marker}:$rc; exec bash')
    pane = ssh('tmux has-session -t hrlrvo && tmux new-window -d -P -F '+
               shlex.quote('#{pane_id}')+' -t hrlrvo -n '+shlex.quote(a.name)+' '+
               shlex.quote('bash -lc '+shlex.quote(command))).strip()
    print(f'pane={pane} log={log}', flush=True)
    if not a.wait:
        return
    deadline = time.monotonic()+a.timeout
    while time.monotonic() < deadline:
        output = ssh('tmux capture-pane -p -S -100 -t '+shlex.quote(pane))
        if marker+':' in output:
            print(output, flush=True)
            local_log = ROOT/'verification'/f'{tag}.log'
            subprocess.run(['scp', '-P', '2222', *OPTIONS, HOST+':'+log, str(local_log)], check=True)
            code = int(output.split(marker+':', 1)[1].splitlines()[0].strip())
            raise SystemExit(code)
        time.sleep(3)
    raise TimeoutError(f'Task remains active in {pane}; do not blindly relaunch: {log}')

if __name__ == '__main__':
    main()
