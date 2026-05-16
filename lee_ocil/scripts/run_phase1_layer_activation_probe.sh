#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SERVER_ID="${SERVER_ID:-$(hostname)}"
SERVER_IP="${SERVER_IP:-unknown}"
GPU="${GPU:-0}"
DATASET_PRED_PAIRS="${DATASET_PRED_PAIRS:-Solar1:96 Solar5:96 ETTh1:24}"
SEEDS="${SEEDS:-2024 2025 2026}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PYTHON="${PYTHON:-.venv/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-phase1_layer_activation_probe_${RUN_ID}}"
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
  local config_name="$3"
  local activation="$4"
  local encoder_activation="$5"
  local decoder_activation="$6"
  local output_activation="$7"
  local amplitude="$8"
  local lee_type="$9"
  local seed="${10}"
  local target
  target="$(target_for_dataset "${dataset}")"
  local log_file="logs/${dataset}_pl${pred_len}_${config_name}_seed${seed}_${SERVER_ID}_${RUN_TAG}.log"

  "${PYTHON}" -u main_informer.py \
    --data "${dataset}" \
    --features M \
    --target "${target}" \
    --pred_len "${pred_len}" \
    --seq_len 96 \
    --label_len 48 \
    --activation "${activation}" \
    --encoder_activation "${encoder_activation}" \
    --decoder_activation "${decoder_activation}" \
    --output_activation "${output_activation}" \
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

for pair in ${DATASET_PRED_PAIRS}; do
  dataset="${pair%%:*}"
  pred_len="${pair##*:}"
  for seed in ${SEEDS}; do
    run_one "${dataset}" "${pred_len}" gelu_all_outlinear gelu gelu gelu linear 0.0 1 "${seed}"
    run_one "${dataset}" "${pred_len}" tanh_all_outlinear tanh tanh tanh linear 0.0 1 "${seed}"
    run_one "${dataset}" "${pred_len}" enc_tanh_dec_gelu_outlinear gelu tanh gelu linear 0.0 1 "${seed}"
    run_one "${dataset}" "${pred_len}" enc_gelu_dec_tanh_outlinear gelu gelu tanh linear 0.0 1 "${seed}"
    run_one "${dataset}" "${pred_len}" tanh_all_outtanh tanh tanh tanh tanh 0.0 1 "${seed}"
    run_one "${dataset}" "${pred_len}" tanh_sin001_all_outlinear tanh_sin tanh_sin tanh_sin linear 0.01 1 "${seed}"
  done
done
