#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

INTERVAL_SECONDS="${INTERVAL_SECONDS:-1200}"
RUN_IDS="${RUN_IDS:-}"
LOG_FILE="${LOG_FILE:-logs/phase1_heartbeat_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p logs
printf "%s\n" "${LOG_FILE}" > logs/current_phase1_heartbeat_log.txt

while true; do
  {
    echo "===== heartbeat $(date -Is) ====="
    echo "-- gpu --"
    nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,name --format=csv,noheader || true
    echo "-- processes --"
    ps -eo pid,ppid,etime,args \
      | awk '/main_informer.py|run_phase1|tanh_specificity|oscillation|heartbeat/ && !/awk/ {print}' \
      || true
    echo "-- current run ids --"
    for file in \
      logs/current_phase1_tanh_specificity_run_id.txt \
      logs/current_phase1_tanh_specificity_solar_run_id.txt \
      logs/current_phase1_oscillation_run_id.txt \
      logs/current_phase1_oscillation_solar_run_id.txt; do
      printf "%s=" "${file}"
      cat "${file}" 2>/dev/null || true
      echo
    done
    echo "-- summary counts --"
    ids="${RUN_IDS}"
    for file in logs/current_phase1_*run_id.txt; do
      if [[ -f "${file}" ]]; then
        id="$(cat "${file}" 2>/dev/null || true)"
        if [[ -n "${id}" ]]; then
          ids="${ids} ${id}"
        fi
      fi
    done
    for id in ${ids}; do
      printf "%s " "${id}"
      grep -c "${id}" results/summary.csv 2>/dev/null || true
    done | sort -u
    echo
  } >> "${LOG_FILE}" 2>&1
  sleep "${INTERVAL_SECONDS}"
done
