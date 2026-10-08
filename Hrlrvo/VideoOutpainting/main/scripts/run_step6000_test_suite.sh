#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/main"
readonly INFERENCE_ENTRYPOINT="${REPO_ROOT}/src/ltx_outpaint/inference.py"
readonly CONFIG_PATH="${REPO_ROOT}/configs/inference_4k.yaml"
readonly CHECKPOINT_DIR="/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/runs/hrlrvo_main_train_v1/checkpoints/checkpoint-006000"
readonly ADAPTER_PATH="${CHECKPOINT_DIR}/adapter_model.safetensors"
readonly CHECKPOINT_MANIFEST="${CHECKPOINT_DIR}/outpainting_manifest.json"
readonly INPUT_ROOT="/home/work/ai-graphics_33/Hrlrvo/VideoOutpainting/input/test_set"
readonly OUTPUT_ROOT="${REPO_ROOT}/outputs/step6000_seed333333_test_suite"
readonly SUITE_LOG="${OUTPUT_ROOT}/run_suite.log"
readonly TIMING_SUMMARY="${OUTPUT_ROOT}/run_time_summary.csv"
readonly SEED=333333
readonly CONDA_ENV_NAME="Hrlrvo"
readonly VERIFIED_CONDA_SH="/home/work/ai-graphics_33/miniconda3/etc/profile.d/conda.sh"

# The current production CLI accepts only --prompt "" and sample_long_range()
# rejects every non-empty prompt. Keep the requested text for auditability, but
# never pass it to an unsupported/fake CLI path.
readonly PROMPT_SUPPORTED=0
readonly REQUESTED_PROMPT="Seamlessly extend the existing concert stage into the outpainted area. Continue the same stage floor, LED screens, lighting, truss, walls, and set design as a natural continuation of the visible stage. Keep the outpainted area as stage environment only, not backstage, not off-stage space, and do not introduce staff, crew, camera operators, audience, cameras, or additional people."

readonly -a BASELINE_INPUTS=(
  "${INPUT_ROOT}/0ad61bb2-a2c7-4e23-b5fa-266edf6b06fa.mp4"
  "${INPUT_ROOT}/0f2313f5-5a61-4ccc-811c-f3bbf1d41151.mp4"
  "${INPUT_ROOT}/b78e2011-b17e-4d55-a28e-862d58187799.mp4"
  "${INPUT_ROOT}/ef334007-aa0b-432d-a7e8-76a3eeeec252.mp4"
  "${INPUT_ROOT}/fea48c69-5ce4-441f-9eaf-4b8bf195bb09.mp4"
)

readonly -a BOOST_INPUTS=(
  "${INPUT_ROOT}/ef334007-aa0b-432d-a7e8-76a3eeeec252.mp4"
  "${INPUT_ROOT}/0f2313f5-5a61-4ccc-811c-f3bbf1d41151.mp4"
  "${INPUT_ROOT}/0ad61bb2-a2c7-4e23-b5fa-266edf6b06fa.mp4"
)

readonly PROMPT_INPUT="${INPUT_ROOT}/fea48c69-5ce4-441f-9eaf-4b8bf195bb09.mp4"

mkdir -p "${OUTPUT_ROOT}"
touch "${SUITE_LOG}"
exec > >(tee -a "${SUITE_LOG}") 2>&1

if [[ ! -s "${TIMING_SUMMARY}" ]]; then
  printf '%s\n' \
    'recorded_at_utc,group,clip_id,early_global_boost,prompt_enabled,start_utc,end_utc,elapsed_seconds,elapsed_human,status,exit_status' \
    > "${TIMING_SUMMARY}"
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
  local prompt_enabled="$4"
  local start_utc="$5"
  local end_utc="$6"
  local elapsed_seconds="$7"
  local elapsed_human="$8"
  local status="$9"
  local exit_status="${10}"

  printf '"%s","%s","%s","%s","%s","%s","%s","%s","%s","%s","%s"\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" \
    "${group}" \
    "${clip_id}" \
    "${boost}" \
    "${prompt_enabled}" \
    "${start_utc}" \
    "${end_utc}" \
    "${elapsed_seconds}" \
    "${elapsed_human}" \
    "${status}" \
    "${exit_status}" \
    >> "${TIMING_SUMMARY}"
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
  append_summary \
    "suite" "all" "mixed" "blocked" \
    "${SUITE_START_UTC}" "${suite_end_utc}" \
    "${suite_elapsed}" "${suite_human}" "${suite_status}" "${exit_status}"
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
  for command_name in python ffprobe tee; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
      log "ERROR: required command is unavailable in ${CONDA_ENV_NAME}: ${command_name}"
      return 1
    fi
  done

  checkpoint_preflight

  local input_path
  for input_path in "${BASELINE_INPUTS[@]}"; do
    if [[ ! -s "${input_path}" ]]; then
      log "ERROR: required input video is missing or empty: ${input_path}"
      return 1
    fi
  done
  if [[ ! -s "${PROMPT_INPUT}" ]]; then
    log "ERROR: prompt-test input video is missing or empty: ${PROMPT_INPUT}"
    return 1
  fi

  log "CONDA ENV: ${CONDA_DEFAULT_ENV}"
  log "PYTHON: $(command -v python)"
  log "CONFIG: ${CONFIG_PATH}"
  log "CHECKPOINT: ${CHECKPOINT_DIR}"
  log "ADAPTER: ${ADAPTER_PATH}"
  log "SEED: ${SEED}"
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
    "manifest_gaps": manifest.get("prefix_gap_count") == 0,
    "manifest_duplicates": manifest.get("prefix_duplicate_count") == 0,
    "lineage_gaps": lineage.get("gap_count") == 0,
    "lineage_duplicates": lineage.get("duplicate_count") == 0,
    "lineage_chunks": isinstance(lineage.get("chunks"), list) and bool(lineage["chunks"]),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("completion contract mismatch: " + ", ".join(failed))
PY
  then
    return 1
  fi

  ffprobe \
    -v error \
    -select_streams v:0 \
    -show_entries stream=codec_type \
    -of default=noprint_wrappers=1:nokey=1 \
    "${output_path}" \
    | grep -qx 'video'
}

run_case() {
  local group="$1"
  local input_path="$2"
  local boost_enabled="$3"
  local clip_id
  clip_id="$(basename -- "${input_path}" .mp4)"
  local run_dir="${OUTPUT_ROOT}/${group}/${clip_id}"
  local output_path="${run_dir}/output.mp4"
  local run_log="${run_dir}/run.log"
  mkdir -p "${run_dir}"

  if completed_output_is_valid "${input_path}" "${output_path}" "${boost_enabled}"; then
    local skipped_at
    skipped_at="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    log "SKIP: group=${group} clip=${clip_id}; completed output contract is valid"
    append_summary \
      "${group}" "${clip_id}" "${boost_enabled}" "false" \
      "${skipped_at}" "${skipped_at}" "0" "0h 00m 00s" "SKIPPED" "0"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    return 0
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

  local start_epoch start_utc end_epoch end_utc elapsed elapsed_human exit_status
  start_epoch="$(date +%s)"
  start_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

  {
    printf '%s\n' "============================================================"
    printf 'GROUP: %s\n' "${group}"
    printf 'CLIP_ID: %s\n' "${clip_id}"
    printf 'INPUT: %s\n' "${input_path}"
    printf 'OUTPUT: %s\n' "${output_path}"
    printf 'CHECKPOINT: %s\n' "${CHECKPOINT_DIR}"
    printf 'SEED: %s\n' "${SEED}"
    printf 'EARLY_GLOBAL_BOOST: %s\n' "${boost_enabled}"
    printf 'PROMPT_ENABLED: false\n'
    printf 'START: %s\n' "${start_utc}"
    printf 'COMMAND:'
    printf ' %q' "${command[@]}"
    printf '\n'
  } | tee -a "${run_log}"

  set +e
  "${command[@]}" 2>&1 | tee -a "${run_log}"
  exit_status=${PIPESTATUS[0]}
  set -e

  end_epoch="$(date +%s)"
  end_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  elapsed=$((end_epoch - start_epoch))
  elapsed_human="$(human_duration "${elapsed}")"

  {
    printf 'END: %s\n' "${end_utc}"
    printf 'ELAPSED SECONDS: %s\n' "${elapsed}"
    printf 'ELAPSED: %s\n' "${elapsed_human}"
    printf 'EXIT STATUS: %s\n' "${exit_status}"
    printf '%s\n' "============================================================"
  } | tee -a "${run_log}"

  RUN_COUNT=$((RUN_COUNT + 1))
  if ((exit_status != 0)); then
    append_summary \
      "${group}" "${clip_id}" "${boost_enabled}" "false" \
      "${start_utc}" "${end_utc}" "${elapsed}" "${elapsed_human}" "FAILED" "${exit_status}"
    log "ERROR: inference failed for group=${group} clip=${clip_id} exit_status=${exit_status}"
    return "${exit_status}"
  fi

  if ! completed_output_is_valid "${input_path}" "${output_path}" "${boost_enabled}"; then
    append_summary \
      "${group}" "${clip_id}" "${boost_enabled}" "false" \
      "${start_utc}" "${end_utc}" "${elapsed}" "${elapsed_human}" "INVALID_OUTPUT" "70"
    log "ERROR: inference exited successfully but its output contract is incomplete or mismatched: ${run_dir}"
    return 70
  fi

  append_summary \
    "${group}" "${clip_id}" "${boost_enabled}" "false" \
    "${start_utc}" "${end_utc}" "${elapsed}" "${elapsed_human}" "COMPLETED" "0"
  log "COMPLETE: group=${group} clip=${clip_id} elapsed=${elapsed_human}"
}

main() {
  log "SUITE START: ${SUITE_START_UTC}"
  preflight

  log "GROUP A START: baseline (5 sequential runs)"
  local input_path
  for input_path in "${BASELINE_INPUTS[@]}"; do
    run_case "baseline" "${input_path}" "false"
  done

  log "GROUP B START: early_global_boost (3 sequential runs)"
  for input_path in "${BOOST_INPUTS[@]}"; do
    run_case "early_global_boost" "${input_path}" "true"
  done

  if ((PROMPT_SUPPORTED)); then
    # This branch is intentionally unreachable for the currently verified
    # production interface. Do not add a prompt CLI until production supports it.
    log "ERROR: internal script contract mismatch: prompt execution was not implemented."
    return 71
  fi

  log "PROMPT TEST BLOCKED: Current inference implementation does not support a runtime text prompt."
  log "PROMPT INPUT: ${PROMPT_INPUT}"
  log "REQUESTED PROMPT (NOT SENT): ${REQUESTED_PROMPT}"
  append_summary \
    "prompt_stage_only" "$(basename -- "${PROMPT_INPUT}" .mp4)" "false" "false" \
    "" "" "0" "0h 00m 00s" "BLOCKED_UNSUPPORTED_PROMPT" "0"

  log "All supported test cases completed successfully."
}

main "$@"
