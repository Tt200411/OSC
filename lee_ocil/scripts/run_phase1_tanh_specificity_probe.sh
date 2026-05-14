#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SERVER_ID="${SERVER_ID:-$(hostname)}"
SERVER_IP="${SERVER_IP:-unknown}"
GPU="${GPU:-0}"
DATASETS="${DATASETS:-ETTh1 ETTh2}"
PRED_LENS="${PRED_LENS:-24 168}"
SEEDS="${SEEDS:-2024 2025 2026}"
TANH_SIN_AMPLITUDES="${TANH_SIN_AMPLITUDES:-0.01 0.05 0.1}"
LEE_TYPES="${LEE_TYPES:-1}"
INCLUDE_GELU_BASELINE="${INCLUDE_GELU_BASELINE:-0}"
INCLUDE_TANH_BASELINE="${INCLUDE_TANH_BASELINE:-1}"
INCLUDE_TANH_SIN="${INCLUDE_TANH_SIN:-1}"
INCLUDE_LEE="${INCLUDE_LEE:-1}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PYTHON="${PYTHON:-.venv/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-phase1_tanh_specificity_${RUN_ID}}"
mkdir -p logs

target_for_dataset() {
  case "$1" in
    Solar|Solar1|Solar5) echo "Power" ;;
    *) echo "OT" ;;
  esac
}

run_one() {
  local dataset="$1"
  local pred_len="$2"
  local activation="$3"
  local amplitude="$4"
  local lee_type="$5"
  local seed="$6"
  local target
  target="$(target_for_dataset "${dataset}")"
  local log_file="logs/${dataset}_pl${pred_len}_${activation}_a${amplitude}_lee${lee_type}_seed${seed}_${SERVER_ID}_${RUN_TAG}.log"

  "${PYTHON}" -u main_informer.py \
    --data "${dataset}" \
    --features M \
    --target "${target}" \
    --pred_len "${pred_len}" \
    --seq_len 96 \
    --label_len 48 \
    --activation "${activation}" \
    --perturb_amplitude "${amplitude}" \
    --perturb_frequency 1.0 \
    --perturb_phase 0.0 \
    --lee_type "${lee_type}" \
    --seed "${seed}" \
    --gpu "${GPU}" \
    --server_id "${SERVER_ID}" \
    --server_ip "${SERVER_IP}" \
    --run_id "${RUN_ID}" \
    --train_epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --des "${RUN_TAG}" 2>&1 | tee "${log_file}"
}

for dataset in ${DATASETS}; do
  for pred_len in ${PRED_LENS}; do
    for seed in ${SEEDS}; do
      if [[ "${INCLUDE_GELU_BASELINE}" == "1" ]]; then
        run_one "${dataset}" "${pred_len}" gelu 0.0 1 "${seed}"
      fi
      if [[ "${INCLUDE_TANH_BASELINE}" == "1" ]]; then
        run_one "${dataset}" "${pred_len}" tanh 0.0 1 "${seed}"
      fi
      if [[ "${INCLUDE_TANH_SIN}" == "1" ]]; then
        for amplitude in ${TANH_SIN_AMPLITUDES}; do
          run_one "${dataset}" "${pred_len}" tanh_sin "${amplitude}" 1 "${seed}"
        done
      fi
      if [[ "${INCLUDE_LEE}" == "1" ]]; then
        for lee_type in ${LEE_TYPES}; do
          run_one "${dataset}" "${pred_len}" lee 0.0 "${lee_type}" "${seed}"
        done
      fi
    done
  done
done
