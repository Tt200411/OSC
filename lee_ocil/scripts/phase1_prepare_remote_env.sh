#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
LOG_FILE="${LOG_FILE:-logs/phase1_prepare_remote_env_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p logs
printf "%s\n" "${LOG_FILE}" > logs/current_phase1_prepare_env_log.txt

{
  echo "===== prepare env $(date -Is) ====="
  echo "python_bootstrap=${PYTHON_BOOTSTRAP}"
  echo "venv_dir=${VENV_DIR}"
  echo "torch_index_url=${TORCH_INDEX_URL}"

  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${PYTHON_BOOTSTRAP}" -m venv "${VENV_DIR}"
  fi

  "${VENV_DIR}/bin/python" -m pip install --upgrade pip wheel setuptools
  "${VENV_DIR}/bin/python" -m pip install numpy pandas tqdm

  if ! "${VENV_DIR}/bin/python" - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    "${VENV_DIR}/bin/python" -m pip install torch --index-url "${TORCH_INDEX_URL}"
  fi

  "${VENV_DIR}/bin/python" - <<'PY'
import sys
import numpy
import pandas
import torch

print("python", sys.version.split()[0])
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
  nvidia-smi --query-gpu=memory.used,utilization.gpu,name --format=csv,noheader || true
  echo "===== prepare env done $(date -Is) ====="
} 2>&1 | tee -a "${LOG_FILE}"
