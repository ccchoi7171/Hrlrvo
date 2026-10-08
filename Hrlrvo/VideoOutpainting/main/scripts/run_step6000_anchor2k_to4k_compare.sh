#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/main"
readonly INFERENCE_ENTRYPOINT="${REPO_ROOT}/src/ltx_outpaint/inference.py"
readonly CONFIG_PATH="${REPO_ROOT}/configs/inference_4k.yaml"
readonly CHECKPOINT_DIR="/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/runs/hrlrvo_main_train_v1/checkpoints/checkpoint-006000"
readonly ADAPTER_PATH="${CHECKPOINT_DIR}/adapter_model.safetensors"
readonly CHECKPOINT_MANIFEST="${CHECKPOINT_DIR}/outpainting_manifest.json"
readonly INPUT_ROOT="/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/input/anchor"
readonly OUTPUT_ROOT="${REPO_ROOT}/outputs/step6000_seed333333_anchor2k_to4k_compare_v1"
readonly SUITE_LOG="${OUTPUT_ROOT}/run_suite.log"
readonly SUMMARY="${OUTPUT_ROOT}/run_summary.csv"
readonly SEED=333333
readonly CONDA_ENV_NAME="Hrlrvo"
readonly VERIFIED_CONDA_SH="/home/work/ai-graphics_33/miniconda3/etc/profile.d/conda.sh"
readonly BBOX_FILTER="drawbox=x=960:y=540:w=1920:h=1080:color=0x00ff00:t=8"

readonly -a BASELINE_INPUTS=(
  "${INPUT_ROOT}/0ad61bb2-a2c7-4e23-b5fa-266edf6b06fa.mp4"
  "${INPUT_ROOT}/0f2313f5-5a61-4ccc-811c-f3bbf1d41151.mp4"
  "${INPUT_ROOT}/ef334007-aa0b-432d-a7e8-76a3eeeec252.mp4"
)

readonly -a BOOST_INPUTS=(
  "${INPUT_ROOT}/0ad61bb2-a2c7-4e23-b5fa-266edf6b06fa.mp4"
  "${INPUT_ROOT}/0f2313f5-5a61-4ccc-811c-f3bbf1d41151.mp4"
  "${INPUT_ROOT}/ef334007-aa0b-432d-a7e8-76a3eeeec252.mp4"
)

mkdir -p "${OUTPUT_ROOT}"
touch "${SUITE_LOG}"
exec > >(tee -a "${SUITE_LOG}") 2>&1

if [[ ! -s "${SUMMARY}" ]]; then
  printf '%s\n' \
    'recorded_at_utc,group,clip_id,early_global_boost,input,output,boxed_output,checkpoint,seed,start_utc,end_utc,inference_elapsed_seconds,inference_elapsed_human,overlay_elapsed_seconds,peak_cuda_allocated_gib,peak_cuda_reserved_gib,status,exit_status' \
    > "${SUMMARY}"
fi

SUITE_START_EPOCH="$(date +%s)"
SUITE_START_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
RUN_COUNT=0
SKIP_COUNT=0

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

human_duration() {
  local total_seconds="$1"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf '%dh %02dm %02ds' "${hours}" "${minutes}" "${seconds}"
}

append_summary() {
  local group="$1"
  local clip_id="$2"
  local boost="$3"
  local input_path="$4"
  local output_path="$5"
  local boxed_path="$6"
  local start_utc="$7"
  local end_utc="$8"
  local inference_elapsed="$9"
  local inference_human="${10}"
  local overlay_elapsed="${11}"
  local peak_allocated_gib="${12}"
  local peak_reserved_gib="${13}"
  local status="${14}"
  local exit_status="${15}"

  printf '"%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s"\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
    "${group}" "${clip_id}" "${boost}" "${input_path}" "${output_path}" \
    "${boxed_path}" "${CHECKPOINT_DIR}" "${SEED}" "${start_utc}" "${end_utc}" \
    "${inference_elapsed}" "${inference_human}" "${overlay_elapsed}" \
    "${peak_allocated_gib}" "${peak_reserved_gib}" "${status}" "${exit_status}" \
    >> "${SUMMARY}"
}

finish_suite() {
  local exit_status=$?
  trap - EXIT
  set +e
  local suite_end_epoch suite_end_utc suite_elapsed suite_human suite_status
  suite_end_epoch="$(date +%s)"
  suite_end_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  suite_elapsed=$((suite_end_epoch - SUITE_START_EPOCH))
  suite_human="$(human_duration "${suite_elapsed}")"
  suite_status="COMPLETED"
  if ((exit_status != 0)); then
    suite_status="FAILED"
  fi

  log "SUITE END: ${suite_end_utc}"
  log "SUITE TOTAL ELAPSED SECONDS: ${suite_elapsed}"
  log "SUITE TOTAL ELAPSED: ${suite_human}"
  log "SUITE STATUS: ${suite_status}"
  log "RUNS EXECUTED: ${RUN_COUNT}"
  log "RUNS SKIPPED: ${SKIP_COUNT}"
  exit "${exit_status}"
}
trap finish_suite EXIT

activate_conda() {
  local conda_sh="${VERIFIED_CONDA_SH}"
  if [[ ! -r "${conda_sh}" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
      log "ERROR: conda was not found and ${VERIFIED_CONDA_SH} is unavailable."
      return 1
    fi
    local conda_base
    conda_base="$(conda info --base)"
    conda_sh="${conda_base}/etc/profile.d/conda.sh"
  fi
  if [[ ! -r "${conda_sh}" ]]; then
    log "ERROR: conda activation script is unavailable: ${conda_sh}"
    return 1
  fi

  set +u
  # shellcheck source=/dev/null
  source "${conda_sh}"
  conda activate "${CONDA_ENV_NAME}"
  set -u

  if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
    log "ERROR: expected conda environment ${CONDA_ENV_NAME}, got ${CONDA_DEFAULT_ENV:-<unset>}."
    return 1
  fi
}

checkpoint_preflight() {
  if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    log "ERROR: checkpoint-006000 is not available yet."
    return 2
  fi
  if [[ ! -s "${ADAPTER_PATH}" || ! -s "${CHECKPOINT_MANIFEST}" ]]; then
    log "ERROR: checkpoint-006000 is not available yet."
    log "Required artifacts: ${ADAPTER_PATH} and ${CHECKPOINT_MANIFEST}"
    return 2
  fi

  python - "${CHECKPOINT_MANIFEST}" "${ADAPTER_PATH}" <<'PY'
import json
import pathlib
import struct
import sys

manifest_path = pathlib.Path(sys.argv[1])
adapter_path = pathlib.Path(sys.argv[2])

with manifest_path.open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)
if int(manifest.get("global_step", -1)) != 6000:
    raise SystemExit(f"checkpoint manifest global_step is not 6000: {manifest.get('global_step')!r}")

with adapter_path.open("rb") as handle:
    raw_length = handle.read(8)
    if len(raw_length) != 8:
        raise SystemExit("adapter safetensors header is truncated")
    header_length = struct.unpack("<Q", raw_length)[0]
    if header_length <= 0 or header_length > 100_000_000:
        raise SystemExit(f"invalid adapter safetensors header length: {header_length}")
    header = json.loads(handle.read(header_length))
metadata = header.get("__metadata__", {})
required = {"architecture", "base_model_path", "provenance"}
missing = sorted(required - set(metadata))
if missing:
    raise SystemExit(f"adapter safetensors metadata is incomplete: missing={missing}")
PY
}

preflight() {
  log "PREFLIGHT START"
  activate_conda
  cd "${REPO_ROOT}"
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

  if [[ ! -f "${INFERENCE_ENTRYPOINT}" ]]; then
    log "ERROR: inference entrypoint is missing: ${INFERENCE_ENTRYPOINT}"
    return 1
  fi
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    log "ERROR: production inference config is missing: ${CONFIG_PATH}"
    return 1
  fi
  for command_name in python ffmpeg ffprobe nvidia-smi tee; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
      log "ERROR: required command is unavailable in ${CONDA_ENV_NAME}: ${command_name}"
      return 1
    fi
  done

  checkpoint_preflight

  python - "${CONFIG_PATH}" "${BASELINE_INPUTS[@]}" <<'PY'
import json
import pathlib
import subprocess
import sys

import yaml

config_path = pathlib.Path(sys.argv[1])
with config_path.open(encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
if config["video"]["canvas"] != [3840, 2160]:
    raise SystemExit("Refusing to run: production canvas is not 3840x2160")
if config["sampling"]["inference_steps"] != 30:
    raise SystemExit("Refusing to run: production inference_steps is not 30")
if config["runtime"]["device"] != "cuda":
    raise SystemExit("Refusing to run: production runtime device is not cuda")
if float(config["video"]["fps"]) != 24.0:
    raise SystemExit("Refusing to run: production FPS is not 24")

for value in sys.argv[2:]:
    path = pathlib.Path(value)
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Required anchor input is missing or empty: {path}")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate,nb_frames", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    if len(streams) != 1:
        raise SystemExit(f"Expected one video stream in {path}")
    video = streams[0]
    found = (video.get("width"), video.get("height"), video.get("avg_frame_rate"))
    if found != (1920, 1080, "24/1") or int(video.get("nb_frames", 0)) < 1:
        raise SystemExit(f"Refusing non-1920x1080, non-24fps, or empty anchor: {path}: {video}")
    print(f"ANCHOR VERIFIED: {path} 1920x1080 24fps {video['nb_frames']} frames", flush=True)
PY

  local gpu_processes
  gpu_processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
  if [[ -n "${gpu_processes}" ]]; then
    log "ERROR: GPU already has compute processes; refusing to start inference: ${gpu_processes}"
    return 1
  fi

  log "CONDA ENV: ${CONDA_DEFAULT_ENV}"
  log "PYTHON: $(command -v python)"
  log "CONFIG: ${CONFIG_PATH}"
  log "CHECKPOINT: ${CHECKPOINT_DIR}"
  log "ADAPTER: ${ADAPTER_PATH}"
  log "SEED: ${SEED}"
  log "GPU COMPUTE PROCESSES: none"
  log "OUTPUT ROOT: ${OUTPUT_ROOT}"
  log "PREFLIGHT PASS"
}

completed_output_is_valid() {
  local input_path="$1"
  local output_path="$2"
  local expected_boost="$3"
  local run_dir
  run_dir="$(dirname -- "${output_path}")"
  local manifest_path="${run_dir}/inference_manifest.json"
  local lineage_path="${run_dir}/chunk_lineage.json"

  if [[ ! -s "${output_path}" || ! -s "${manifest_path}" || ! -s "${lineage_path}" ]]; then
    return 1
  fi

  if ! python - \
    "${manifest_path}" \
    "${lineage_path}" \
    "${input_path}" \
    "${output_path}" \
    "${ADAPTER_PATH}" \
    "${SEED}" \
    "${expected_boost}" <<'PY'
import json
import pathlib
import subprocess
import sys

manifest_path, lineage_path, input_path, output_path, adapter_path = map(pathlib.Path, sys.argv[1:6])
expected_seed = int(sys.argv[6])
expected_boost = sys.argv[7].lower() == "true"

with manifest_path.open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)
with lineage_path.open("r", encoding="utf-8") as handle:
    lineage = json.load(handle)

checks = {
    "input_path": pathlib.Path(manifest.get("input_path", "")).resolve() == input_path.resolve(),
    "output_path": pathlib.Path(manifest.get("output_path", "")).resolve() == output_path.resolve(),
    "seed": manifest.get("seed") == expected_seed,
    "output_finite": manifest.get("output_finite") is True,
    "positive_prompt": manifest.get("fixed_positive_prompt") == "",
    "negative_prompt": manifest.get("fixed_negative_prompt") == "",
    "boost": bool(manifest.get("early_global_boost", {}).get("enabled")) is expected_boost,
    "adapter": pathlib.Path(manifest.get("adapter_load", {}).get("adapter_path", "")).resolve()
    == adapter_path.resolve(),
    "adapter_keys": manifest.get("adapter_load", {}).get("loaded_key_count")
    == manifest.get("adapter_load", {}).get("expected_key_count"),
    "adapter_missing": manifest.get("adapter_load", {}).get("missing_key_count") == 0,
    "adapter_unexpected": manifest.get("adapter_load", {}).get("unexpected_key_count") == 0,
    "adapter_shapes": manifest.get("adapter_load", {}).get("shape_mismatch_count") == 0,
    "anchor_shape": manifest.get("input_rgb_shape", [])[2:] == [1080, 1920],
    "output_shape": manifest.get("output_rgb_shape", [])[2:] == [2160, 3840],
    "vram_allocated": manifest.get("peak_cuda_allocated_bytes", 0) > 0,
    "vram_reserved": manifest.get("peak_cuda_reserved_bytes", 0) > 0,
    "manifest_gaps": manifest.get("prefix_gap_count") == 0,
    "manifest_duplicates": manifest.get("prefix_duplicate_count") == 0,
    "lineage_gaps": lineage.get("gap_count") == 0,
    "lineage_duplicates": lineage.get("duplicate_count") == 0,
    "lineage_chunks": isinstance(lineage.get("chunks"), list) and bool(lineage["chunks"]),
}
if checks["lineage_chunks"]:
    runtime = lineage["chunks"][0].get("runtime", {})
    latent_shape = runtime.get("full_generation_latent_shape", [])
    anchor_rect = runtime.get("known_anchor_rect_yxyx", [])
    checks["actual_outpaint_region"] = (
        len(latent_shape) == 5
        and len(anchor_rect) == 4
        and anchor_rect != [0, 0, latent_shape[3], latent_shape[4]]
        and anchor_rect[0] == latent_shape[3] - anchor_rect[2]
        and anchor_rect[1] == latent_shape[4] - anchor_rect[3]
        and runtime.get("anchor_rgb_shape", [])[2:] == [1080, 1920]
    )

probe = subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
     "stream=width,height,avg_frame_rate,nb_frames", "-of", "json", str(output_path)],
    capture_output=True, text=True, check=True,
)
streams = json.loads(probe.stdout).get("streams", [])
expected_frames = manifest.get("output_rgb_shape", [None])[0]
checks["video_stream"] = (
    len(streams) == 1
    and streams[0].get("width") == 3840
    and streams[0].get("height") == 2160
    and streams[0].get("avg_frame_rate") == "24/1"
    and int(streams[0].get("nb_frames", 0)) == expected_frames
)
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("completion contract mismatch: " + ", ".join(failed))
PY
  then
    return 1
  fi
}

boxed_output_is_valid() {
  local boxed_path="$1"
  local manifest_path="$2"
  [[ -s "${boxed_path}" ]] || return 1
  python - "${boxed_path}" "${manifest_path}" <<'PY'
import json
import pathlib
import subprocess
import sys

boxed = pathlib.Path(sys.argv[1])
manifest = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
probe = subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
     "stream=width,height,avg_frame_rate,nb_frames", "-of", "json", str(boxed)],
    capture_output=True, text=True, check=True,
)
streams = json.loads(probe.stdout).get("streams", [])
if len(streams) != 1:
    raise SystemExit("Boxed output must have exactly one video stream")
video = streams[0]
if (video.get("width"), video.get("height"), video.get("avg_frame_rate")) != (3840, 2160, "24/1"):
    raise SystemExit("Boxed output has wrong size or FPS")
if int(video.get("nb_frames", 0)) != manifest["output_rgb_shape"][0]:
    raise SystemExit("Boxed output frame count differs from inference output")
PY
}

run_case() {
  local group="$1"
  local input_path="$2"
  local boost_enabled="$3"
  local clip_id
  clip_id="$(basename -- "${input_path}" .mp4)"
  local run_dir="${OUTPUT_ROOT}/${group}/${clip_id}"
  local output_path="${run_dir}/output.mp4"
  local boxed_path="${run_dir}/output_anchor_bbox_green.mp4"
  local manifest_path="${run_dir}/inference_manifest.json"
  local lineage_path="${run_dir}/chunk_lineage.json"
  local run_log="${run_dir}/run.log"
  mkdir -p "${run_dir}"

  local start_epoch end_epoch start_utc end_utc inference_elapsed inference_human
  local overlay_elapsed=0 peak_allocated_gib="" peak_reserved_gib=""
  local inference_status="COMPLETED"
  start_epoch="$(date +%s)"
  start_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  if completed_output_is_valid "${input_path}" "${output_path}" "${boost_enabled}"; then
    log "SKIP INFERENCE: group=${group} clip=${clip_id}; completed output contract is valid"
    inference_elapsed=0
    inference_human="$(human_duration 0)"
    inference_status="SKIPPED_INFERENCE"
    SKIP_COUNT=$((SKIP_COUNT + 1))
  else
    if [[ -e "${output_path}" || -e "${manifest_path}" || -e "${lineage_path}" ]]; then
      log "ERROR: incomplete output exists; refusing to overwrite it: ${run_dir}"
      return 70
    fi

    local -a command=(
      python "${INFERENCE_ENTRYPOINT}"
      --config "${CONFIG_PATH}"
      --anchor-video "${input_path}"
      --output "${output_path}"
      --checkpoint "${CHECKPOINT_DIR}"
      --seed "${SEED}"
    )
    if [[ "${boost_enabled}" == "true" ]]; then
      command+=(--early-global-boost)
    fi

    {
      printf '%s\n' "============================================================"
      printf 'GROUP: %s\n' "${group}"
      printf 'CLIP_ID: %s\n' "${clip_id}"
      printf 'INPUT: %s\n' "${input_path}"
      printf 'OUTPUT: %s\n' "${output_path}"
      printf 'BOXED OUTPUT: %s\n' "${boxed_path}"
      printf 'CHECKPOINT: %s\n' "${CHECKPOINT_DIR}"
      printf 'SEED: %s\n' "${SEED}"
      printf 'EARLY_GLOBAL_BOOST: %s\n' "${boost_enabled}"
      printf 'START: %s\n' "${start_utc}"
      printf 'COMMAND:'
      printf ' %q' "${command[@]}"
      printf '\n'
    } | tee -a "${run_log}"

    local exit_status
    set +e
    "${command[@]}" 2>&1 | tee -a "${run_log}"
    exit_status=${PIPESTATUS[0]}
    set -e

    end_epoch="$(date +%s)"
    end_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    inference_elapsed=$((end_epoch - start_epoch))
    inference_human="$(human_duration "${inference_elapsed}")"
    RUN_COUNT=$((RUN_COUNT + 1))

    log "INFERENCE END: group=${group} clip=${clip_id} exit=${exit_status} elapsed=${inference_human}"
    if ((exit_status != 0)); then
      append_summary "${group}" "${clip_id}" "${boost_enabled}" "${input_path}" \
        "${output_path}" "${boxed_path}" "${start_utc}" "${end_utc}" \
        "${inference_elapsed}" "${inference_human}" "0" "" "" "FAILED_INFERENCE" "${exit_status}"
      return "${exit_status}"
    fi
    if ! completed_output_is_valid "${input_path}" "${output_path}" "${boost_enabled}"; then
      log "ERROR: inference returned zero but output contract is invalid: ${run_dir}"
      append_summary "${group}" "${clip_id}" "${boost_enabled}" "${input_path}" \
        "${output_path}" "${boxed_path}" "${start_utc}" "${end_utc}" \
        "${inference_elapsed}" "${inference_human}" "0" "" "" "INVALID_OUTPUT" "70"
      return 70
    fi
  fi

  read -r peak_allocated_gib peak_reserved_gib < <(python - "${manifest_path}" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
gib = 1024 ** 3
print(f"{manifest['peak_cuda_allocated_bytes'] / gib:.3f} {manifest['peak_cuda_reserved_bytes'] / gib:.3f}")
PY
  )
  log "VRAM PEAK: group=${group} clip=${clip_id} allocated=${peak_allocated_gib} GiB reserved=${peak_reserved_gib} GiB (PyTorch manifest)"

  if boxed_output_is_valid "${boxed_path}" "${manifest_path}"; then
    log "SKIP BOX OVERLAY: valid boxed MP4 already exists: ${boxed_path}"
  else
    if [[ -e "${boxed_path}" ]]; then
      log "ERROR: invalid boxed output exists; refusing to overwrite it: ${boxed_path}"
      return 73
    fi
    local overlay_start overlay_exit temp_path
    overlay_start="$(date +%s)"
    temp_path="${run_dir}/output_anchor_bbox_green.partial.$$.mp4"
    log "BOX OVERLAY START: group=${group} clip=${clip_id} bbox=(960,540)-(2880,1620), green, 8px"
    set +e
    ffmpeg -nostdin -hide_banner -loglevel error -i "${output_path}" \
      -map 0:v:0 -vf "${BBOX_FILTER}" -an \
      -c:v libx264 -preset veryfast -crf 18 -pix_fmt yuv420p -movflags +faststart \
      "${temp_path}" 2>&1 | tee -a "${run_log}"
    overlay_exit=${PIPESTATUS[0]}
    set -e
    overlay_elapsed=$(($(date +%s) - overlay_start))
    if ((overlay_exit != 0)) || ! boxed_output_is_valid "${temp_path}" "${manifest_path}"; then
      log "ERROR: boxed MP4 generation failed: group=${group} clip=${clip_id} exit=${overlay_exit}"
      append_summary "${group}" "${clip_id}" "${boost_enabled}" "${input_path}" \
        "${output_path}" "${boxed_path}" "${start_utc}" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
        "${inference_elapsed}" "${inference_human}" "${overlay_elapsed}" \
        "${peak_allocated_gib}" "${peak_reserved_gib}" "FAILED_OVERLAY" "73"
      return 73
    fi
    mv -n -- "${temp_path}" "${boxed_path}"
    log "BOX OVERLAY COMPLETE: ${boxed_path} elapsed=$(human_duration "${overlay_elapsed}")"
  fi

  end_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  append_summary "${group}" "${clip_id}" "${boost_enabled}" "${input_path}" \
    "${output_path}" "${boxed_path}" "${start_utc}" "${end_utc}" \
    "${inference_elapsed}" "${inference_human}" "${overlay_elapsed}" \
    "${peak_allocated_gib}" "${peak_reserved_gib}" "${inference_status}" "0"
  log "RUN COMPLETE: group=${group} clip=${clip_id} inference=${inference_human} peak_vram=${peak_allocated_gib} GiB"
}

main() {
  log "SUITE START: ${SUITE_START_UTC}"
  preflight

  log "GROUP A START: baseline (3 sequential 1920x1080-to-3840x2160 runs)"
  local input_path
  for input_path in "${BASELINE_INPUTS[@]}"; do
    run_case "baseline" "${input_path}" "false"
  done

  log "GROUP B START: early_global_boost (3 sequential runs)"
  for input_path in "${BOOST_INPUTS[@]}"; do
    run_case "early_global_boost" "${input_path}" "true"
  done

  log "All six inference runs and their green-boxed variants completed successfully."
}

main "$@"
