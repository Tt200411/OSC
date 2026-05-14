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
RUN_TAG="${RUN_TAG:-phase1_activation_ablation_${RUN_ID}}"
mkdir -p logs

run_one() {
  local activation="$1"
  local seed="$2"
  local log_file="logs/ETTh1_pl24_${activation}_seed${seed}_${SERVER_ID}_${RUN_TAG}.log"

  "${PYTHON}" -u main_informer.py \
    --data ETTh1 \
    --features M \
    --target OT \
    --pred_len 24 \
    --seq_len 96 \
    --label_len 48 \
    --activation "${activation}" \
    --perturb_amplitude 0.0 \
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

for seed in ${SEEDS}; do
  run_one gelu "${seed}"
  run_one tanh "${seed}"
  run_one scaled_tanh "${seed}"
  run_one softsign "${seed}"
done
