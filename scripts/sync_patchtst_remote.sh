#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 USER@HOST [REMOTE_DIR]" >&2
  exit 2
fi

REMOTE="$1"
REMOTE_DIR="${2:-~/project/osc_informer}"
SSH_OPTS="${SSH_OPTS:-}"
PASSWORD_ENV="${PATCHTST_SSH_PASSWORD:-${PHASE4_SSH_PASSWORD:-}}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

run_with_auth() {
  if [[ -n "${PASSWORD_ENV}" ]]; then
    export PATCHTST_SSH_PASSWORD="${PASSWORD_ENV}"
    expect -f - "$@" <<'EOF'
set timeout -1
set password $env(PATCHTST_SSH_PASSWORD)
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

run_with_auth ssh ${SSH_OPTS} "${REMOTE}" "mkdir -p ${REMOTE_DIR}/PatchTST ${REMOTE_DIR}/Solar ${REMOTE_DIR}/lee_ocil/ETT-small ${REMOTE_DIR}/lee_ocil/scripts"

run_with_auth rsync -av -e "${RSYNC_RSH}" --delete \
  --exclude '.git/' \
  --exclude 'checkpoints/' \
  --exclude 'results/' \
  --exclude 'logs/' \
  --exclude 'test_results/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "${ROOT_DIR}/PatchTST/PatchTST_supervised/" "${REMOTE}:${REMOTE_DIR}/PatchTST/PatchTST_supervised/"

run_with_auth rsync -av -e "${RSYNC_RSH}" \
  "${ROOT_DIR}/lee_ocil/ETT-small/ETTh2.csv" \
  "${ROOT_DIR}/lee_ocil/ETT-small/ETTm2.csv" \
  "${REMOTE}:${REMOTE_DIR}/lee_ocil/ETT-small/"

run_with_auth rsync -av -e "${RSYNC_RSH}" \
  "${ROOT_DIR}/Solar/Site_2_130MW.csv" \
  "${ROOT_DIR}/Solar/Site_3_30MW.csv" \
  "${ROOT_DIR}/Solar/Site_5_110MW.csv" \
  "${REMOTE}:${REMOTE_DIR}/Solar/"

run_with_auth rsync -av -e "${RSYNC_RSH}" \
  "${ROOT_DIR}/lee_ocil/scripts/generate_factors.py" \
  "${ROOT_DIR}/lee_ocil/scripts/generate_phase4_factors.py" \
  "${ROOT_DIR}/lee_ocil/scripts/analyze_factor_effects.py" \
  "${ROOT_DIR}/lee_ocil/scripts/analyze_regime_activation_oracle.py" \
  "${REMOTE}:${REMOTE_DIR}/lee_ocil/scripts/"

echo "Synced PatchTST files to ${REMOTE}:${REMOTE_DIR}"
