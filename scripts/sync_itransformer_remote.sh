#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 USER@HOST [REMOTE_DIR]" >&2
  exit 2
fi

REMOTE="$1"
REMOTE_DIR="${2:-~/project/osc_informer}"
SSH_OPTS="${SSH_OPTS:-}"
PASSWORD_ENV="${ITRANSFORMER_SSH_PASSWORD:-${PHASE4_SSH_PASSWORD:-}}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

run_with_auth() {
  if [[ -n "${PASSWORD_ENV}" ]]; then
    export ITRANSFORMER_SSH_PASSWORD="${PASSWORD_ENV}"
    expect -f - "$@" <<'EOF'
set timeout -1
set password $env(ITRANSFORMER_SSH_PASSWORD)
set command [lrange $argv 0 end]
log_user 0
spawn {*}$command
set output ""
expect {
  -re "(?i)password:" {
    send -- "$password\r"
    exp_continue
  }
  -re ".+" {
    append output $expect_out(buffer)
    exp_continue
  }
  eof
}
catch wait result
puts -nonewline $output
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

run_with_auth ssh ${SSH_OPTS} "${REMOTE}" "mkdir -p ${REMOTE_DIR}/iTransformer ${REMOTE_DIR}/lee_ocil/ETT-small ${REMOTE_DIR}/lee_ocil/scripts"

run_with_auth rsync -av -e "${RSYNC_RSH}" --delete \
  --exclude '.git/' \
  --exclude 'checkpoints/' \
  --exclude 'results/' \
  --exclude 'test_results/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "${ROOT_DIR}/iTransformer/" "${REMOTE}:${REMOTE_DIR}/iTransformer/"

run_with_auth rsync -av -e "${RSYNC_RSH}" \
  "${ROOT_DIR}/lee_ocil/ETT-small/ETTh2.csv" \
  "${ROOT_DIR}/lee_ocil/ETT-small/ETTm2.csv" \
  "${REMOTE}:${REMOTE_DIR}/lee_ocil/ETT-small/"

run_with_auth rsync -av -e "${RSYNC_RSH}" \
  "${ROOT_DIR}/lee_ocil/scripts/phase4_queue_manager.py" \
  "${REMOTE}:${REMOTE_DIR}/lee_ocil/scripts/"

echo "Synced iTransformer files to ${REMOTE}:${REMOTE_DIR}"
