#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SERVER_ID="${SERVER_ID:-$(hostname)}"
SERVER_IP="${SERVER_IP:-unknown}"
GPU="${GPU:-0}"
DATASETS="${DATASETS:-ETTh1 ETTh2}"
PRED_LENS="${PRED_LENS:-24 168}"
SEEDS="${SEEDS:-2024 2025 2026}"
BASE_ACTIVATIONS="${BASE_ACTIVATIONS:-tanh}"
OSC_ACTIVATIONS="${OSC_ACTIVATIONS:-tanh_sin tanh_cos tanh_rand}"
OSC_AMPLITUDES="${OSC_AMPLITUDES:-0.01 0.1}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PYTHON="${PYTHON:-.venv/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-phase1_oscillation_forms_${RUN_ID}}"
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
  local seed="$5"
  local target
  target="$(target_for_dataset "${dataset}")"
  local log_file="logs/${dataset}_pl${pred_len}_${activation}_a${amplitude}_seed${seed}_${SERVER_ID}_${RUN_TAG}.log"

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
    --lee_type 1 \
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
      for activation in ${BASE_ACTIVATIONS}; do
        run_one "${dataset}" "${pred_len}" "${activation}" 0.0 "${seed}"
      done
      for activation in ${OSC_ACTIVATIONS}; do
        for amplitude in ${OSC_AMPLITUDES}; do
          run_one "${dataset}" "${pred_len}" "${activation}" "${amplitude}" "${seed}"
        done
      done
    done
  done
done
