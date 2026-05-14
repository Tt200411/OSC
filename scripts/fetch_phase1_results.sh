#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 USER@HOST [REMOTE_DIR] [LOCAL_DIR]" >&2
  exit 2
fi

REMOTE="$1"
REMOTE_DIR="${2:-~/project/osc_informer}"
LOCAL_DIR="${3:-phase1_remote_results/$(echo "${REMOTE}" | tr '@:/' '____')}"
SSH_OPTS="${SSH_OPTS:-}"

mkdir -p "${LOCAL_DIR}"

rsync -av -e "ssh ${SSH_OPTS}" \
  "${REMOTE}:${REMOTE_DIR}/lee_ocil/results/" \
  "${LOCAL_DIR}/results/"

rsync -av -e "ssh ${SSH_OPTS}" \
  "${REMOTE}:${REMOTE_DIR}/lee_ocil/logs/" \
  "${LOCAL_DIR}/logs/" || true

echo "Fetched Phase 1 results into ${LOCAL_DIR}"
