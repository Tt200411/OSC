#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

SERVER_ID="${SERVER_ID:-4090-248}"
SERVER_IP="${SERVER_IP:-10.20.12.248}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-.venv/bin/python}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
WAIT_FOR_PIDS="${WAIT_FOR_PIDS:-}"
LOG_FILE="${LOG_FILE:-logs/phase1_auto_research_queue_$(date +%Y%m%d_%H%M%S).log}"

RUN_LEE13_ETT="${RUN_LEE13_ETT:-1}"
RUN_LEE13_SOLAR="${RUN_LEE13_SOLAR:-1}"
RUN_FORMS_ETT="${RUN_FORMS_ETT:-1}"
RUN_FORMS_SOLAR="${RUN_FORMS_SOLAR:-1}"

LEE13_ETT_DATASETS="${LEE13_ETT_DATASETS:-ETTh1 ETTh2}"
LEE13_ETT_PRED_LENS="${LEE13_ETT_PRED_LENS:-24 168}"
LEE13_SOLAR_DATASETS="${LEE13_SOLAR_DATASETS:-Solar1 Solar5}"
LEE13_SOLAR_PRED_LENS="${LEE13_SOLAR_PRED_LENS:-24 96}"
LEE13_SEEDS="${LEE13_SEEDS:-2024 2025 2026}"
LEE13_TANH_SIN_AMPLITUDES="${LEE13_TANH_SIN_AMPLITUDES:-0.01 0.05 0.1}"
LEE13_TYPES="${LEE13_TYPES:-1 3}"

FORMS_ETT_DATASETS="${FORMS_ETT_DATASETS:-ETTh1 ETTh2}"
FORMS_ETT_PRED_LENS="${FORMS_ETT_PRED_LENS:-24 168}"
FORMS_SOLAR_DATASETS="${FORMS_SOLAR_DATASETS:-Solar1 Solar5}"
FORMS_SOLAR_PRED_LENS="${FORMS_SOLAR_PRED_LENS:-24 96}"
FORMS_SEEDS="${FORMS_SEEDS:-2024 2025 2026}"
FORMS_BASE_ACTIVATIONS="${FORMS_BASE_ACTIVATIONS:-tanh}"
FORMS_OSC_ACTIVATIONS="${FORMS_OSC_ACTIVATIONS:-tanh_sin tanh_cos tanh_rand}"
FORMS_OSC_AMPLITUDES="${FORMS_OSC_AMPLITUDES:-0.01 0.1}"

mkdir -p logs
printf "%s\n" "${LOG_FILE}" > logs/current_phase1_auto_research_queue_log.txt

log() {
  printf "[%s] %s\n" "$(date -Is)" "$*" | tee -a "${LOG_FILE}"
}

wait_for_pids() {
  if [[ -z "${WAIT_FOR_PIDS}" ]]; then
    return 0
  fi

  while true; do
    local active=()
    local pid
    for pid in ${WAIT_FOR_PIDS}; do
      if kill -0 "${pid}" 2>/dev/null; then
        active+=("${pid}")
      fi
    done
    if [[ "${#active[@]}" -eq 0 ]]; then
      log "all wait pids finished: ${WAIT_FOR_PIDS}"
      return 0
    fi
    log "waiting for existing training pids: ${active[*]}"
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
}

wait_for_training_idle() {
  while pgrep -f "main_informer.py" >/dev/null 2>&1; do
    log "main_informer.py still active; waiting before starting next queued block"
    sleep "${CHECK_INTERVAL_SECONDS}"
  done
}

run_block() {
  local name="$1"
  shift
  log "start ${name}: $*"
  if "$@" 2>&1 | tee -a "${LOG_FILE}"; then
    log "done ${name}"
  else
    local status=$?
    log "failed ${name} status=${status}; continuing to next block"
  fi
}

run_tanh_specificity_lee13_ett() {
  local run_id="phase1_tanh_specificity_lee13_ett_${SERVER_ID}_$(date +%Y%m%d_%H%M%S)"
  printf "%s\n" "${run_id}" > logs/current_phase1_tanh_specificity_lee13_ett_run_id.txt
  env SERVER_ID="${SERVER_ID}" SERVER_IP="${SERVER_IP}" GPU="${GPU}" \
    DATASETS="${LEE13_ETT_DATASETS}" PRED_LENS="${LEE13_ETT_PRED_LENS}" SEEDS="${LEE13_SEEDS}" \
    TANH_SIN_AMPLITUDES="${LEE13_TANH_SIN_AMPLITUDES}" LEE_TYPES="${LEE13_TYPES}" \
    INCLUDE_GELU_BASELINE=0 INCLUDE_TANH_BASELINE=1 INCLUDE_TANH_SIN=1 INCLUDE_LEE=1 \
    EPOCHS="${EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PYTHON="${PYTHON}" \
    RUN_ID="${run_id}" RUN_TAG="${run_id}" \
    bash scripts/run_phase1_tanh_specificity_probe.sh
}

run_tanh_specificity_lee13_solar() {
  local run_id="phase1_tanh_specificity_lee13_solar_${SERVER_ID}_$(date +%Y%m%d_%H%M%S)"
  printf "%s\n" "${run_id}" > logs/current_phase1_tanh_specificity_lee13_solar_run_id.txt
  env SERVER_ID="${SERVER_ID}" SERVER_IP="${SERVER_IP}" GPU="${GPU}" \
    DATASETS="${LEE13_SOLAR_DATASETS}" PRED_LENS="${LEE13_SOLAR_PRED_LENS}" SEEDS="${LEE13_SEEDS}" \
    TANH_SIN_AMPLITUDES="${LEE13_TANH_SIN_AMPLITUDES}" LEE_TYPES="${LEE13_TYPES}" \
    INCLUDE_GELU_BASELINE=0 INCLUDE_TANH_BASELINE=1 INCLUDE_TANH_SIN=1 INCLUDE_LEE=1 \
    EPOCHS="${EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PYTHON="${PYTHON}" \
    RUN_ID="${run_id}" RUN_TAG="${run_id}" \
    bash scripts/run_phase1_tanh_specificity_probe.sh
}

run_oscillation_forms_ett() {
  local run_id="phase1_oscillation_forms_ett_${SERVER_ID}_$(date +%Y%m%d_%H%M%S)"
  printf "%s\n" "${run_id}" > logs/current_phase1_oscillation_forms_ett_run_id.txt
  env SERVER_ID="${SERVER_ID}" SERVER_IP="${SERVER_IP}" GPU="${GPU}" \
    DATASETS="${FORMS_ETT_DATASETS}" PRED_LENS="${FORMS_ETT_PRED_LENS}" SEEDS="${FORMS_SEEDS}" \
    BASE_ACTIVATIONS="${FORMS_BASE_ACTIVATIONS}" OSC_ACTIVATIONS="${FORMS_OSC_ACTIVATIONS}" OSC_AMPLITUDES="${FORMS_OSC_AMPLITUDES}" \
    EPOCHS="${EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PYTHON="${PYTHON}" \
    RUN_ID="${run_id}" RUN_TAG="${run_id}" \
    bash scripts/run_phase1_oscillation_forms_probe.sh
}

run_oscillation_forms_solar() {
  local run_id="phase1_oscillation_forms_solar_${SERVER_ID}_$(date +%Y%m%d_%H%M%S)"
  printf "%s\n" "${run_id}" > logs/current_phase1_oscillation_forms_solar_run_id.txt
  env SERVER_ID="${SERVER_ID}" SERVER_IP="${SERVER_IP}" GPU="${GPU}" \
    DATASETS="${FORMS_SOLAR_DATASETS}" PRED_LENS="${FORMS_SOLAR_PRED_LENS}" SEEDS="${FORMS_SEEDS}" \
    BASE_ACTIVATIONS="${FORMS_BASE_ACTIVATIONS}" OSC_ACTIVATIONS="${FORMS_OSC_ACTIVATIONS}" OSC_AMPLITUDES="${FORMS_OSC_AMPLITUDES}" \
    EPOCHS="${EPOCHS}" BATCH_SIZE="${BATCH_SIZE}" PYTHON="${PYTHON}" \
    RUN_ID="${run_id}" RUN_TAG="${run_id}" \
    bash scripts/run_phase1_oscillation_forms_probe.sh
}

log "auto research queue started"
wait_for_pids
wait_for_training_idle

if [[ "${RUN_LEE13_ETT}" == "1" ]]; then
  run_block "tanh specificity lee1/lee3 ETT" run_tanh_specificity_lee13_ett
fi

if [[ "${RUN_LEE13_SOLAR}" == "1" ]]; then
  run_block "tanh specificity lee1/lee3 Solar1/Solar5" run_tanh_specificity_lee13_solar
fi

if [[ "${RUN_FORMS_ETT}" == "1" ]]; then
  run_block "oscillation forms ETT" run_oscillation_forms_ett
fi

if [[ "${RUN_FORMS_SOLAR}" == "1" ]]; then
  run_block "oscillation forms Solar1/Solar5" run_oscillation_forms_solar
fi

log "auto research queue finished"
