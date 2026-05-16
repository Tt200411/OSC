#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 USER@HOST [REMOTE_DIR]" >&2
  exit 2
fi

REMOTE="$1"
REMOTE_DIR="${2:-~/project/osc_informer}"
SSH_OPTS="${SSH_OPTS:-}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

ssh ${SSH_OPTS} "${REMOTE}" "mkdir -p ${REMOTE_DIR}/Solar ${REMOTE_DIR}/lee_ocil/ETT-small"

rsync -av -e "ssh ${SSH_OPTS}" --delete \
  --exclude 'checkpoints/' \
  --exclude 'results/' \
  --exclude 'logs/' \
  --exclude '.venv*/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'ETT-small/*.csv' \
  "${ROOT_DIR}/lee_ocil/" "${REMOTE}:${REMOTE_DIR}/lee_ocil/"

rsync -av -e "ssh ${SSH_OPTS}" \
  "${ROOT_DIR}/lee_ocil/ETT-small/ETTh1.csv" \
  "${ROOT_DIR}/lee_ocil/ETT-small/ETTh2.csv" \
  "${REMOTE}:${REMOTE_DIR}/lee_ocil/ETT-small/"

rsync -av -e "ssh ${SSH_OPTS}" \
  "${ROOT_DIR}/Solar/PV_Solar_Station_1.csv" \
  "${REMOTE}:${REMOTE_DIR}/Solar/PV_Solar_Station_1.csv"

rsync -av -e "ssh ${SSH_OPTS}" \
  "${ROOT_DIR}/Solar/Site_1_50MW.csv" \
  "${ROOT_DIR}/Solar/Site_5_110MW.csv" \
  "${REMOTE}:${REMOTE_DIR}/Solar/"

echo "Synced core Phase 1 files to ${REMOTE}:${REMOTE_DIR}"
