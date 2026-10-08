# Hrlrvo Read-only Training Monitor

A standalone Python/Dash dashboard that incrementally reads `train_metrics.jsonl`. It does not import the Training Python package, communicate with the Training process, inspect checkpoints, use CUDA, or expose any start/stop/resume/write operation.

## Isolation contract

- Web project and all of its data live under `/home/work/ai-graphics_33/Hrlrvo/monitor_Web`.
- Training input is read-only and limited to discovered `runs/*/train_metrics.jsonl` plus sibling `train_console.log`, `run_config.yaml`, and `run_manifest.json`.
- The monitor uses filesystem polling only. There is no IPC/RPC to Training.
- JSONL files are opened only with `rb`; console/config/manifest files are opened only for reading.
- The application never loads PyTorch, CUDA, LTX, checkpoints, or Training source.

## Environment

Create once:

```bash
cd /home/work/ai-graphics_33/Hrlrvo/monitor_Web
conda env create -f environment.yml
conda activate monitor_web
```

The environment already exists on this host. To update an existing environment without deleting it:

```bash
cd /home/work/ai-graphics_33/Hrlrvo/monitor_Web
conda env update -n monitor_web -f environment.yml --prune
conda activate monitor_web
```

Equivalent pip dependencies are pinned in `requirements.txt`.

## Run

### Auto-discover Training runs

```bash
python -B app.py \
  --runs-root /home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/runs \
  --host 0.0.0.0 \
  --port 8050
```

The selector discovers `/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/runs/*/train_metrics.jsonl`, newest first. Its labels show run name, file update time, and latest optimizer step. Newly created run directories appear on the next discovery poll.

### Direct log

`--log` must be absolute and takes precedence over discovery:

```bash
python -B app.py \
  --log /absolute/path/to/train_metrics.jsonl \
  --host 0.0.0.0 \
  --port 8050
```

### Mock dashboard

```bash
python -B app.py --mock --host 0.0.0.0 --port 8050
```

Mock mode reads only `mock_data/` and does not inspect `VideoOutpainting`. Regenerate the deterministic 160-step mock if desired:

```bash
python -B scripts/generate_mock.py --steps 160
```

Open `http://localhost:8050`. From another machine, use `http://<server-address>:8050` subject to the host firewall/network policy.

## tmux

```bash
tmux new -s hrlrvo_monitor
cd /home/work/ai-graphics_33/Hrlrvo/monitor_Web
conda activate monitor_web
python -B app.py --runs-root /home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/runs --host 0.0.0.0 --port 8050
```

Detach with `Ctrl-b d`. Reattach with `tmux attach -t hrlrvo_monitor`. Stopping this Web process has no effect on Training.

## Dashboard panels

- Status bar: run, optimizer step, samples, batches, loss, LR, logging status
- Loss: raw and Web-computed EMA (`alpha=0.1`)
- Learning rate
- LoRA gradients: OP-LoRA, A-LoRA, G-LoRA, F-LoRA
- Compression/restoration gradients: C2, **Shared C4**, R2, R4
- Fusion gradients: Local Fusion and Global Fusion
- Local actual fusion ratio: Native and Adaptive histories plus latest mean/min/max
- Global actual fusion ratio: Local and Global histories plus latest mean/min/max
- 48-block Local and Global fusion distributions with optimizer-step selection
- Training health: global norm, NaN, Inf, zero-gradient groups, frozen-gradient violations, clipping before/after
- Current gradient table and normalized **Gradient Magnitude** chart
- Read-only tail of `train_console.log` (50/100/200/500 lines)
- Collapsible read-only run config and manifest

Missing schema fields render as `N/A`; they do not crash the dashboard. Common alternative logger names such as `global_step`, `train_loss`, `lr`, and `*_grad_norm` are normalized.

## Incremental reader behavior

Each selected log has its own in-memory reader state:

- Maintains the last byte offset and reads only newly appended bytes.
- Retains an incomplete final line until a newline arrives.
- Skips and counts malformed completed lines.
- Detects inode replacement, truncation, and rewritten file prefixes, then safely reloads.
- Retains up to 100,000 records per run by default; override with `--max-records`.
- Does not acquire a file lock or rotate/truncate the source.

Logging status uses the newest usable JSON timestamp/file mtime: `RUNNING <= 10 s`, `STALE <= 60 s`, otherwise `STOPPED / NO UPDATE`. It never probes or signals a Training PID.

## Test

```bash
python -B -m pytest -q
```

Coverage includes mock parsing, incomplete appends, malformed lines, missing fields, NaN/Inf health, 48-block arrays, run switching, incremental tailing, replacement/truncation, missing console logs, Dash layout/routes, and callback rendering.

Health endpoints for smoke checks:

```bash
curl -fsS http://127.0.0.1:8050/healthz
curl -fsS http://127.0.0.1:8050/api/snapshot
```
