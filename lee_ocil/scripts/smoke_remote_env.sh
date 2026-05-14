#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"

echo "== python =="
"${PYTHON}" -V

echo "== pip =="
"${PYTHON}" -m pip --version || true

echo "== torch/cuda =="
"${PYTHON}" - <<'PY'
import json
try:
    import torch
    payload = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ],
    }
except Exception as exc:
    payload = {"torch_import_error": repr(exc)}
print(json.dumps(payload, indent=2))
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "== nvidia-smi =="
  nvidia-smi
else
  echo "nvidia-smi not found"
fi

echo "== ETTh1 dataloader smoke =="
"${PYTHON}" -u main_informer.py \
  --data ETTh1 \
  --features M \
  --pred_len 24 \
  --activation gelu \
  --batch_size 4 \
  --use_gpu false \
  --smoke_test

echo "== Solar dataloader smoke =="
"${PYTHON}" -u main_informer.py \
  --data Solar \
  --features M \
  --pred_len 24 \
  --activation gelu \
  --batch_size 4 \
  --use_gpu false \
  --smoke_test
