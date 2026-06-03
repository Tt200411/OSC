#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SERVER_ID="${SERVER_ID:-$(hostname)}"
SERVER_IP="${SERVER_IP:-unknown}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-.venv/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-phase2_original_protocol_${RUN_ID}}"
SEEDS="${SEEDS:-2024 2025 2026}"
PLAN="${PLAN:-ett_core}"
CONFIGS="${CONFIGS:-gelu_all tanh_all softsign_all lee1_all lee3_all}"
DATASET_FILTER="${DATASET_FILTER:-ALL}"
PRED_FILTER="${PRED_FILTER:-ALL}"
EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DROPOUT="${DROPOUT:-0.05}"
EMBED="${EMBED:-timeF}"
SAVE_ARRAYS="${SAVE_ARRAYS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
mkdir -p logs results

common_flags() {
  printf '%s ' \
    --model informer \
    --features M \
    --d_model 512 \
    --n_heads 8 \
    --d_ff 2048 \
    --attn prob \
    --embed "${EMBED}" \
    --distil true \
    --mix true \
    --batch_size "${BATCH_SIZE}" \
    --learning_rate 1e-4 \
    --patience 3 \
    --lradj type1 \
    --train_epochs "${EPOCHS}" \
    --dropout "${DROPOUT}"
}

target_for_dataset() {
  case "$1" in
    Solar|Solar1|Solar5) echo "Power" ;;
    *) echo "OT" ;;
  esac
}

freq_for_dataset() {
  case "$1" in
    Solar|Solar1|Solar5) echo "t" ;;
    *) echo "h" ;;
  esac
}

dataset_config_rows() {
  case "${PLAN}" in
    ett_core)
      cat <<'ROWS'
ETTh1 168 168 168 2 1 5
ETTh1 336 168 168 2 1 5
ETTh2 168 336 336 3 2 5
ETTh2 336 336 168 3 2 5
ROWS
      ;;
    ett_short)
      cat <<'ROWS'
ETTh1 24 48 48 2 1 3
ETTh1 48 96 48 2 1 5
ETTh2 24 48 48 2 1 5
ETTh2 48 96 96 2 1 5
ROWS
      ;;
    ett_long720)
      cat <<'ROWS'
ETTh1 720 336 336 2 1 5
ETTh2 720 720 336 3 2 5
ROWS
      ;;
    solar_core)
      cat <<'ROWS'
Solar1 24 96 48 2 1 5
Solar1 96 96 48 2 1 5
Solar5 24 96 48 2 1 5
Solar5 96 96 48 2 1 5
ROWS
      ;;
    solar_96)
      cat <<'ROWS'
Solar1 96 96 48 2 1 5
Solar5 96 96 48 2 1 5
ROWS
      ;;
    *)
      echo "Unknown PLAN=${PLAN}" >&2
      exit 2
      ;;
  esac
}

config_flags() {
  case "$1" in
    gelu_all)
      echo "--activation gelu --encoder_activation gelu --decoder_activation gelu --output_activation linear --perturb_amplitude 0 --lee_type 1"
      ;;
    relu_all)
      echo "--activation relu --encoder_activation relu --decoder_activation relu --output_activation linear --perturb_amplitude 0 --lee_type 1"
      ;;
    swish_all)
      echo "--activation swish --encoder_activation swish --decoder_activation swish --output_activation linear --perturb_amplitude 0 --lee_type 1"
      ;;
    tanh_all)
      echo "--activation tanh --encoder_activation tanh --decoder_activation tanh --output_activation linear --perturb_amplitude 0 --lee_type 1"
      ;;
    softsign_all)
      echo "--activation softsign --encoder_activation softsign --decoder_activation softsign --output_activation linear --perturb_amplitude 0 --lee_type 1"
      ;;
    lee1_all)
      echo "--activation lee --encoder_activation lee --decoder_activation lee --output_activation linear --perturb_amplitude 0 --lee_type 1"
      ;;
    lee3_all)
      echo "--activation lee --encoder_activation lee --decoder_activation lee --output_activation linear --perturb_amplitude 0 --lee_type 3"
      ;;
    enc_tanh_dec_gelu)
      echo "--activation gelu --encoder_activation tanh --decoder_activation gelu --output_activation linear --perturb_amplitude 0 --lee_type 1"
      ;;
    enc_gelu_dec_tanh)
      echo "--activation gelu --encoder_activation gelu --decoder_activation tanh --output_activation linear --perturb_amplitude 0 --lee_type 1"
      ;;
    tanh_all_outtanh)
      echo "--activation tanh --encoder_activation tanh --decoder_activation tanh --output_activation tanh --perturb_amplitude 0 --lee_type 1"
      ;;
    gelu_all_outtanh)
      echo "--activation gelu --encoder_activation gelu --decoder_activation gelu --output_activation tanh --perturb_amplitude 0 --lee_type 1"
      ;;
    tanh_sin001_all)
      echo "--activation tanh_sin --encoder_activation tanh_sin --decoder_activation tanh_sin --output_activation linear --perturb_amplitude 0.01 --perturb_frequency 1.0 --perturb_phase 0.0 --lee_type 1"
      ;;
    *)
      echo "Unknown config: $1" >&2
      exit 2
      ;;
  esac
}

run_one() {
  local dataset="$1"
  local pred_len="$2"
  local seq_len="$3"
  local label_len="$4"
  local e_layers="$5"
  local d_layers="$6"
  local factor="$7"
  local config_name="$8"
  local seed="$9"
  local target freq extra_flags log_file save_flags
  target="$(target_for_dataset "${dataset}")"
  freq="$(freq_for_dataset "${dataset}")"
  extra_flags="$(config_flags "${config_name}")"
  log_file="logs/${RUN_TAG}_${dataset}_pl${pred_len}_${config_name}_seed${seed}_${SERVER_ID}.log"
  save_flags=""
  if [[ "${SAVE_ARRAYS}" != "1" ]]; then
    save_flags="--no_save_pred --no_save_true"
  fi
  if [[ "${SKIP_EXISTING}" == "1" && -f results/summary.csv ]]; then
    if grep -F ",${dataset},${pred_len}," results/summary.csv | grep -F ",${seed}," | grep -F "_${PLAN}_${config_name}," >/dev/null 2>&1; then
      echo "===== $(date -Is) skip existing ${dataset} pred=${pred_len} ${config_name} seed=${seed} ====="
      return 0
    fi
  fi

  echo "===== $(date -Is) ${dataset} pred=${pred_len} seq=${seq_len} label=${label_len} ${config_name} seed=${seed} ====="
  # shellcheck disable=SC2086
  "${PYTHON}" -u main_informer.py \
    $(common_flags) \
    --data "${dataset}" \
    --target "${target}" \
    --freq "${freq}" \
    --detail_freq "${freq}" \
    --seq_len "${seq_len}" \
    --label_len "${label_len}" \
    --pred_len "${pred_len}" \
    --e_layers "${e_layers}" \
    --d_layers "${d_layers}" \
    --factor "${factor}" \
    ${extra_flags} \
    --seed "${seed}" \
    --gpu "${GPU}" \
    --server_id "${SERVER_ID}" \
    --server_ip "${SERVER_IP}" \
    --run_id "${RUN_ID}" \
    --des "${RUN_TAG}_${PLAN}_${config_name}" \
    ${save_flags} 2>&1 | tee "${log_file}"
}

while read -r dataset pred_len seq_len label_len e_layers d_layers factor; do
  [[ -z "${dataset}" ]] && continue
  if [[ "${DATASET_FILTER}" != "ALL" && "${DATASET_FILTER}" != "${dataset}" ]]; then
    continue
  fi
  if [[ "${PRED_FILTER}" != "ALL" && " ${PRED_FILTER} " != *" ${pred_len} "* ]]; then
    continue
  fi
  for seed in ${SEEDS}; do
    for config_name in ${CONFIGS}; do
      run_one "${dataset}" "${pred_len}" "${seq_len}" "${label_len}" "${e_layers}" "${d_layers}" "${factor}" "${config_name}" "${seed}"
    done
  done
done < <(dataset_config_rows)

echo "Completed ${RUN_TAG} plan=${PLAN}"
