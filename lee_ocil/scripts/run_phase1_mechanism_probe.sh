#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SERVER_ID="${SERVER_ID:-$(hostname)}"
SERVER_IP="${SERVER_IP:-unknown}"
GPU="${GPU:-0}"
DATASETS="${DATASETS:-ETTh2 Solar}"
PRED_LEN="${PRED_LEN:-24}"
SEEDS="${SEEDS:-2024}"
ACTIVATIONS="${ACTIVATIONS:-gelu lee tanh softsign}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PYTHON="${PYTHON:-.venv/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-phase1_mechanism_probe_${RUN_ID}}"
mkdir -p logs

target_for_dataset() {
  case "$1" in
    Solar) echo "Power" ;;
    *) echo "OT" ;;
  esac
}

run_one() {
  local dataset="$1"
  local activation="$2"
  local seed="$3"
  local target
  target="$(target_for_dataset "${dataset}")"
  local log_file="logs/${dataset}_pl${PRED_LEN}_${activation}_seed${seed}_${SERVER_ID}_${RUN_TAG}.log"

  "${PYTHON}" -u main_informer.py \
    --data "${dataset}" \
    --features M \
    --target "${target}" \
    --pred_len "${PRED_LEN}" \
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

for dataset in ${DATASETS}; do
  for seed in ${SEEDS}; do
    for activation in ${ACTIVATIONS}; do
      run_one "${dataset}" "${activation}" "${seed}"
    done
  done
done
