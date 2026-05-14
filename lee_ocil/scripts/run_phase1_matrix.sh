#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SERVER_ID="${SERVER_ID:-$(hostname)}"
SERVER_IP="${SERVER_IP:-unknown}"
GPU="${GPU:-0}"
SEEDS="${SEEDS:-2024 2025 2026}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PYTHON="${PYTHON:-.venv/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-phase1_matrix_${RUN_ID}}"
mkdir -p logs

run_one() {
  local dataset="$1"
  local pred_len="$2"
  local activation="$3"
  local amplitude="$4"
  local lee_type="$5"
  local seed="$6"

  local target="OT"
  local seq_len="96"
  local label_len="48"
  if [[ "${dataset}" == "Solar" ]]; then
    target="Power"
  fi
  if [[ "${pred_len}" -ge 168 ]]; then
    seq_len="${pred_len}"
    label_len="${pred_len}"
  fi
  if [[ "${pred_len}" -ge 336 ]]; then
    label_len="336"
  fi
  local log_file="logs/${dataset}_pl${pred_len}_${activation}_a${amplitude}_lee${lee_type}_seed${seed}_${SERVER_ID}_${RUN_TAG}.log"

  "${PYTHON}" -u main_informer.py \
    --data "${dataset}" \
    --features M \
    --target "${target}" \
    --pred_len "${pred_len}" \
    --seq_len "${seq_len}" \
    --label_len "${label_len}" \
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

for dataset in ETTh1 ETTh2; do
  for pred_len in 24 48 168 336 720; do
    for seed in ${SEEDS}; do
      run_one "${dataset}" "${pred_len}" gelu 0.0 1 "${seed}"
      run_one "${dataset}" "${pred_len}" relu 0.0 1 "${seed}"
      for amplitude in 0.01 0.05 0.1; do
        run_one "${dataset}" "${pred_len}" gelu_sin "${amplitude}" 1 "${seed}"
        run_one "${dataset}" "${pred_len}" relu_sin "${amplitude}" 1 "${seed}"
      done
      for lee_type in 1 2 3 4; do
        run_one "${dataset}" "${pred_len}" lee 0.0 "${lee_type}" "${seed}"
      done
      run_one "${dataset}" "${pred_len}" dynamic_gelu_sin 0.05 1 "${seed}"
    done
  done
done

for dataset in Solar; do
  for pred_len in 24 48 96 192; do
    for seed in ${SEEDS}; do
      run_one "${dataset}" "${pred_len}" gelu 0.0 1 "${seed}"
      run_one "${dataset}" "${pred_len}" relu 0.0 1 "${seed}"
      for amplitude in 0.01 0.05 0.1; do
        run_one "${dataset}" "${pred_len}" gelu_sin "${amplitude}" 1 "${seed}"
        run_one "${dataset}" "${pred_len}" relu_sin "${amplitude}" 1 "${seed}"
      done
      for lee_type in 1 2 3 4; do
        run_one "${dataset}" "${pred_len}" lee 0.0 "${lee_type}" "${seed}"
      done
      run_one "${dataset}" "${pred_len}" dynamic_gelu_sin 0.05 1 "${seed}"
    done
  done
done
