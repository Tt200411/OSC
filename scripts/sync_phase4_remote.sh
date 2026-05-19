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
SSH_ARGS=()
if [[ -n "${SSH_OPTS}" ]]; then
  # shellcheck disable=SC2206
  SSH_ARGS=(${SSH_OPTS})
fi

run_with_auth() {
  if [[ -n "${PHASE4_SSH_PASSWORD:-}" ]]; then
    expect <<'EOF' "${PHASE4_SSH_PASSWORD}" "$@"
set timeout -1
set password [lindex $argv 0]
set command [lrange $argv 1 end]
spawn {*}$command
expect {
  -re "(?i)password:" {
    send -- "$password\r"
    exp_continue
  }
  eof
}
catch wait result
exit [lindex $result 3]
EOF
  else
    "$@"
  fi
}

RSYNC_RSH="ssh"
if [[ -n "${SSH_OPTS}" ]]; then
  RSYNC_RSH="ssh ${SSH_OPTS}"
fi

run_with_auth ssh "${SSH_ARGS[@]}" "${REMOTE}" "mkdir -p ${REMOTE_DIR}/Solar ${REMOTE_DIR}/lee_ocil/ETT-small"

run_with_auth rsync -av -e "${RSYNC_RSH}" --delete \
  --exclude 'checkpoints/' \
  --exclude 'results/' \
  --exclude 'logs/' \
  --exclude '.venv*/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'ETT-small/*.csv' \
  "${ROOT_DIR}/lee_ocil/" "${REMOTE}:${REMOTE_DIR}/lee_ocil/"

run_with_auth rsync -av -e "${RSYNC_RSH}" \
  "${ROOT_DIR}/lee_ocil/ETT-small/ETTh1.csv" \
  "${ROOT_DIR}/lee_ocil/ETT-small/ETTh2.csv" \
  "${ROOT_DIR}/lee_ocil/ETT-small/ETTm1.csv" \
  "${ROOT_DIR}/lee_ocil/ETT-small/ETTm2.csv" \
  "${REMOTE}:${REMOTE_DIR}/lee_ocil/ETT-small/"

run_with_auth rsync -av -e "${RSYNC_RSH}" \
  "${ROOT_DIR}/Solar/PV_Solar_Station_1.csv" \
  "${ROOT_DIR}/Solar/Site_1_50MW.csv" \
  "${ROOT_DIR}/Solar/Site_2_130MW.csv" \
  "${ROOT_DIR}/Solar/Site_3_30MW.csv" \
  "${ROOT_DIR}/Solar/Site_4_130MW.csv" \
  "${ROOT_DIR}/Solar/Site_5_110MW.csv" \
  "${ROOT_DIR}/Solar/Site_6_35MW.csv" \
  "${ROOT_DIR}/Solar/Site_7_30MW.csv" \
  "${ROOT_DIR}/Solar/Site_8_30MW.csv" \
  "${REMOTE}:${REMOTE_DIR}/Solar/"

echo "Synced Phase-4 files to ${REMOTE}:${REMOTE_DIR}"
