#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting
MAIN="$ROOT/main"
CONFIG="$MAIN/configs/inference_4k.yaml"
CHECKPOINT="$ROOT/runs/hrlrvo_main_train_v1/checkpoints/checkpoint-006000"
INPUT_ROOT=/home/work/ai-graphics_33/Hrlrvo/KBS-OutPaint_Data/Data/10sec_2K_24fps
OUT_ROOT="$ROOT/outputs/prompt_ablation_20260917"
SEED=33333333
POSITIVE='A seamless continuation of the same concert stage, with matching stage design, lighting, LED screens, floor, and backdrop.'
NEGATIVE='backstage, off-stage area, staff, crew, camera operators, production equipment, cables, equipment cases, venue corridors'

source /home/work/ai-graphics_33/miniconda3/etc/profile.d/conda.sh
conda activate Hrlrvo
cd "$MAIN"
export PYTHONPATH="$MAIN/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

test -f "$CONFIG"
test -f "$CHECKPOINT/adapter_model.safetensors"
test -f "$CHECKPOINT/outpainting_manifest.json"
test -f "$INPUT_ROOT/01_RIIZE.mp4"
test -f "$INPUT_ROOT/01_NewJeans_How_Sweet.mp4"

python - "$CONFIG" "$CHECKPOINT" "$INPUT_ROOT" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
manifest = json.loads((Path(sys.argv[2]) / "outpainting_manifest.json").read_text())
assert manifest["global_step"] == 6000 and manifest["architecture_version"] == 8
assert config["model"]["base_model_path"] == manifest["base_model_path"]
assert config["video"]["canvas"] == [3840, 2160]
assert config["video"]["fps"] == 24.0
assert config["sampling"].get("negative_prompt", "") == ""
assert config["sampling"]["seed"] == 33333333
assert config["sampling"]["inference_steps"] == 30
for name in ("01_RIIZE.mp4", "01_NewJeans_How_Sweet.mp4"):
    path = Path(sys.argv[3]) / name
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,avg_frame_rate,duration,nb_frames", "-of", "json", str(path),
    ], text=True))["streams"][0]
    assert (probe["width"], probe["height"]) == (1920, 1080)
    assert probe["avg_frame_rate"] == "24/1"
    assert int(probe["nb_frames"]) == 240
    assert abs(float(probe["duration"]) - 10.0) < 0.01
PY

mkdir -p "$OUT_ROOT"
exec 9>"$OUT_ROOT/.suite.lock"
flock -n 9 || { echo 'Another prompt ablation suite is running' >&2; exit 1; }
cp -n "$CONFIG" "$OUT_ROOT/config_used.yaml"
SUMMARY="$OUT_ROOT/run_summary.csv"
SUITE_LOG="$OUT_ROOT/run_suite.log"
if [[ ! -e "$SUMMARY" ]]; then
    printf 'test_name,input,prompt_enabled,negative_enabled,start_utc,end_utc,elapsed_seconds,status,output\n' > "$SUMMARY"
fi

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log() { printf '[%s] %s\n' "$(stamp)" "$*" | tee -a "$SUITE_LOG"; }

verify_result() {
    local output="$1" input="$2" positive="$3" negative="$4"
    python - "$output" "$input" "$positive" "$negative" "$CHECKPOINT" "$SEED" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

output, source, positive, negative, checkpoint, seed = sys.argv[1:]
video = Path(output)
manifest_path = video.parent / "inference_manifest.json"
lineage = video.parent / "chunk_lineage.json"
if not video.is_file() or not manifest_path.is_file() or not lineage.is_file():
    raise SystemExit("Missing output video or inference manifest/lineage")
manifest = json.loads(manifest_path.read_text())
if (manifest.get("input_path") != str(Path(source).resolve())
        or manifest.get("output_path") != str(video.resolve())
        or manifest.get("positive_prompt") != positive
        or manifest.get("negative_prompt") != negative
        or manifest.get("seed") != int(seed)
        or manifest.get("early_global_boost", {}).get("enabled") is not False
        or manifest.get("adapter_load", {}).get("adapter_path")
            != str(Path(checkpoint).resolve() / "adapter_model.safetensors")
        or manifest.get("prefix_gap_count") != 0
        or manifest.get("prefix_duplicate_count") != 0):
    raise SystemExit("Inference manifest does not match requested experiment")
data = json.loads(subprocess.check_output([
    "ffprobe", "-v", "error", "-count_frames", "-show_entries",
    "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,duration,nb_read_frames:format=duration",
    "-of", "json", str(video),
], text=True))
streams = data["streams"]
if len(streams) != 1 or streams[0]["codec_type"] != "video":
    raise SystemExit("Output must contain one video stream and no audio")
stream = streams[0]
if ((stream["width"], stream["height"]) != (3840, 2160)
        or stream["avg_frame_rate"] != "24/1"
        or stream["r_frame_rate"] != "24/1"
        or int(stream["nb_read_frames"]) != 240
        or abs(float(stream.get("duration") or data["format"]["duration"]) - 10.0) > 1 / 24):
    raise SystemExit(f"Output media mismatch: {stream}")
print(json.dumps({"width": stream["width"], "height": stream["height"],
                  "fps": stream["avg_frame_rate"], "frames": int(stream["nb_read_frames"]),
                  "duration": stream.get("duration") or data["format"]["duration"],
                  "inference_wall_time_seconds": manifest.get("wall_time_seconds")}, sort_keys=True))
PY
}

run_test() {
    local test_name="$1" input="$2" positive="$3" negative="$4"
    local run_dir="$OUT_ROOT/$test_name" output="$OUT_ROOT/$test_name/outpaint.mp4"
    mkdir -p "$run_dir"
    local -a cmd=(python src/ltx_outpaint/inference.py
        --config "$CONFIG" --anchor-video "$input" --output "$output"
        --checkpoint "$CHECKPOINT" --seed "$SEED"
        --prompt "$positive" --negative-prompt "$negative")
    printf '%q ' "${cmd[@]}" > "$run_dir/command.sh"
    printf '\n' >> "$run_dir/command.sh"
    if verify_result "$output" "$input" "$positive" "$negative" >/dev/null 2>&1; then
        log "SKIP_VALID $test_name output=$output"
        return 0
    fi

    local start_iso end_iso start_epoch end_epoch elapsed rc
    start_iso="$(stamp)"
    start_epoch="$(date +%s)"
    log "START $test_name input=$input output=$output checkpoint=$CHECKPOINT seed=$SEED prompt=$(printf '%q' "$positive") negative_prompt=$(printf '%q' "$negative") config=$CONFIG early_global_boost=OFF"
    log "COMMAND $test_name $(cat "$run_dir/command.sh")"
    set +e
    "${cmd[@]}" > "$run_dir/run.log" 2>&1
    rc=$?
    set -e
    end_iso="$(stamp)"
    end_epoch="$(date +%s)"
    elapsed=$((end_epoch - start_epoch))
    if (( rc != 0 )); then
        log "FAIL $test_name exit=$rc elapsed_seconds=$elapsed log=$run_dir/run.log"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$test_name" "$input" "$([[ -n $positive ]] && echo yes || echo no)" "$([[ -n $negative ]] && echo yes || echo no)" "$start_iso" "$end_iso" "$elapsed" "FAIL_$rc" "$output" >> "$SUMMARY"
        return "$rc"
    fi
    local probe_summary
    if ! probe_summary="$(verify_result "$output" "$input" "$positive" "$negative")"; then
        log "FAIL_VALIDATION $test_name elapsed_seconds=$elapsed log=$run_dir/run.log"
        printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$test_name" "$input" "$([[ -n $positive ]] && echo yes || echo no)" "$([[ -n $negative ]] && echo yes || echo no)" "$start_iso" "$end_iso" "$elapsed" "FAIL_VALIDATION" "$output" >> "$SUMMARY"
        return 1
    fi
    printf '%s\n' "$probe_summary" > "$run_dir/ffprobe_summary.json"
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$test_name" "$input" "$([[ -n $positive ]] && echo yes || echo no)" "$([[ -n $negative ]] && echo yes || echo no)" "$start_iso" "$end_iso" "$elapsed" "PASS" "$output" >> "$SUMMARY"
    log "PASS $test_name elapsed_seconds=$elapsed media=$probe_summary output=$output"
}

log "SUITE_START output_root=$OUT_ROOT config_sha256=$(sha256sum "$CONFIG" | cut -d' ' -f1) checkpoint=$CHECKPOINT seed=$SEED"
run_test 01_RIIZE_no_prompt "$INPUT_ROOT/01_RIIZE.mp4" '' ''
run_test 02_RIIZE_with_prompt "$INPUT_ROOT/01_RIIZE.mp4" "$POSITIVE" "$NEGATIVE"
run_test 03_NewJeans_no_prompt "$INPUT_ROOT/01_NewJeans_How_Sweet.mp4" '' ''
run_test 04_NewJeans_with_prompt "$INPUT_ROOT/01_NewJeans_How_Sweet.mp4" "$POSITIVE" "$NEGATIVE"
log "SUITE_COMPLETE four tests passed"
