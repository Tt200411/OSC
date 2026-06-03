#!/usr/bin/env bash
set -euo pipefail

HOSTS_DEFAULT="10.21.53.62 10.21.53.82 10.21.53.113 10.21.53.142 10.21.53.162"
DISABLED_HOSTS="10.20.12.248 10.20.12.247"
REMOTE_USER="${REMOTE_USER:-testsv}"
REMOTE_DIR="${REMOTE_DIR:-~/project/osc_informer}"
HOSTS="${HOSTS:-${HOSTS_DEFAULT}}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-.aris/patchtst_remote_${RUN_TS}}"
QUEUE_MANAGER="${QUEUE_MANAGER:-lee_ocil/scripts/phase4_queue_manager.py}"
PASSWORD_ENV="${PATCHTST_SSH_PASSWORD:-${PHASE4_SSH_PASSWORD:-}}"

mkdir -p "${OUT_DIR}"

ssh_cmd() {
  local host="$1"
  shift
  if [[ -n "${PASSWORD_ENV}" ]]; then
    export PATCHTST_SSH_PASSWORD="${PASSWORD_ENV}"
    expect -f - "${REMOTE_USER}" "${host}" "$@" <<'EOF'
set timeout -1
set user [lindex $argv 0]
set host [lindex $argv 1]
set password $env(PATCHTST_SSH_PASSWORD)
set args [lrange $argv 2 end]
log_user 0
spawn ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$user@$host" {*}$args
set output ""
expect {
  -re "(?i)password:" { send -- "$password\r"; exp_continue }
  -re ".+" { append output $expect_out(buffer); exp_continue }
  eof
}
catch wait result
puts -nonewline $output
exit [lindex $result 3]
EOF
  else
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${REMOTE_USER}@${host}" "$@"
  fi
}

scp_cmd() {
  local source="$1"
  local host="$2"
  local dest="$3"
  if [[ -n "${PASSWORD_ENV}" ]]; then
    export PATCHTST_SSH_PASSWORD="${PASSWORD_ENV}"
    expect -f - "${source}" "${REMOTE_USER}" "${host}" "${dest}" <<'EOF'
set timeout -1
set source [lindex $argv 0]
set user [lindex $argv 1]
set host [lindex $argv 2]
set dest [lindex $argv 3]
set password $env(PATCHTST_SSH_PASSWORD)
log_user 0
spawn scp -o StrictHostKeyChecking=accept-new "$source" "$user@$host:$dest"
set output ""
expect {
  -re "(?i)password:" { send -- "$password\r"; exp_continue }
  -re ".+" { append output $expect_out(buffer); exp_continue }
  eof
}
catch wait result
puts -nonewline $output
exit [lindex $result 3]
EOF
  else
    scp "${source}" "${REMOTE_USER}@${host}:${dest}"
  fi
}

rsync_pull_cmd() {
  local host="$1"
  local source="$2"
  local dest="$3"
  shift 3
  mkdir -p "${dest}"
  if [[ -n "${PASSWORD_ENV}" ]]; then
    export PATCHTST_SSH_PASSWORD="${PASSWORD_ENV}"
    expect -f - "${REMOTE_USER}" "${host}" "${source}" "${dest}" "$@" <<'EOF'
set timeout -1
set user [lindex $argv 0]
set host [lindex $argv 1]
set source [lindex $argv 2]
set dest [lindex $argv 3]
set rsync_opts [lrange $argv 4 end]
set password $env(PATCHTST_SSH_PASSWORD)
log_user 0
spawn rsync -av {*}$rsync_opts -e "ssh -o StrictHostKeyChecking=accept-new" "$user@$host:$source" "$dest"
set output ""
expect {
  -re "(?i)password:" { send -- "$password\r"; exp_continue }
  -re ".+" { append output $expect_out(buffer); exp_continue }
  eof
}
catch wait result
puts -nonewline $output
exit [lindex $result 3]
EOF
  else
    rsync -av "$@" -e "ssh -o BatchMode=yes" "${REMOTE_USER}@${host}:${source}" "${dest}"
  fi
}

inventory() {
  local path="${OUT_DIR}/patchtst_gpu_inventory_${RUN_TS}.md"
  {
    echo "# PatchTST GPU Inventory"
    echo
    echo "Generated: $(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo
    echo "Disabled hosts: ${DISABLED_HOSTS}"
    echo
    echo "| host | status | gpu | memory.used | memory.total | patchtst | data_5_5 | notes |"
    echo "| --- | --- | --- | --- | --- | --- | --- | --- |"
  } > "${path}"
  for host in ${HOSTS}; do
    case " ${DISABLED_HOSTS} " in *" ${host} "*) continue ;; esac
    set +e
    output="$(ssh_cmd "${host}" "cd ${REMOTE_DIR} && gpu=\$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || true); ok=0; for f in lee_ocil/ETT-small/ETTh2.csv lee_ocil/ETT-small/ETTm2.csv Solar/Site_2_130MW.csv Solar/Site_3_30MW.csv Solar/Site_5_110MW.csv; do [ -f \"\$f\" ] && ok=\$((ok+1)); done; [ -f PatchTST/PatchTST_supervised/run_longExp.py ] && pt=yes || pt=no; printf '%s\n%s\n%s\n' \"\$gpu\" \"\$pt\" \"\$ok\"" 2>&1)"
    rc=$?
    set -e
    if [[ "${rc}" -ne 0 ]]; then
      note="${output//$'\n'/ }"
      echo "| ${host} | ssh_failed |  |  |  | no | 0/5 | ${note} |" >> "${path}"
      continue
    fi
    output="$(printf '%s\n' "${output}" | tr -d '\r' | sed '/^[[:space:]]*$/d')"
    gpu_line="$(printf '%s\n' "${output}" | sed -n '1p')"
    patchtst="$(printf '%s\n' "${output}" | sed -n '2p')"
    data_count="$(printf '%s\n' "${output}" | sed -n '3p')"
    IFS=',' read -r gpu_name mem_used mem_total <<< "${gpu_line}"
    echo "| ${host} | ok | ${gpu_name} | ${mem_used} MiB | ${mem_total} MiB | ${patchtst} | ${data_count}/5 |  |" >> "${path}"
  done
  echo "${path}"
}

sync_all() {
  for host in ${HOSTS}; do
    case " ${DISABLED_HOSTS} " in *" ${host} "*) continue ;; esac
    PATCHTST_SSH_PASSWORD="${PASSWORD_ENV:-}" scripts/sync_patchtst_remote.sh "${REMOTE_USER}@${host}" "${REMOTE_DIR}"
  done
}

launch_queues() {
  local manifest_dir="${1:?manifest dir required}"
  for host in ${HOSTS}; do
    case " ${DISABLED_HOSTS} " in *" ${host} "*) continue ;; esac
    local manifest="${manifest_dir}/patchtst_queue_manifest_${host//./_}.json"
    [[ -f "${manifest}" ]] || continue
    local remote_rel=".aris_queue/patchtst/${RUN_TS}_${host//./_}"
    ssh_cmd "${host}" "mkdir -p \"\$HOME/${remote_rel}/logs\" \"\$HOME/.aris_queue\""
    scp_cmd "${QUEUE_MANAGER}" "${host}" ".aris_queue/patchtst_queue_manager.py"
    scp_cmd "${manifest}" "${host}" "${remote_rel}/manifest.json"
    local screen_name="PTQM_${RUN_TS}_${host//./_}"
    ssh_cmd "${host}" "screen -S ${screen_name} -X quit >/dev/null 2>&1 || true; screen -dmS ${screen_name} bash -lc 'cd ${REMOTE_DIR}/PatchTST/PatchTST_supervised && /home/testsv/project/osc_informer/lee_ocil/.venv/bin/python \"\$HOME/.aris_queue/patchtst_queue_manager.py\" --manifest \"\$HOME/${remote_rel}/manifest.json\" --state \"\$HOME/${remote_rel}/queue_state.json\" --log-dir \"\$HOME/${remote_rel}/logs\" > \"\$HOME/${remote_rel}/queue_mgr.log\" 2>&1'"
    echo "${host} ${remote_rel}" >> "${OUT_DIR}/patchtst_launched_queues_${RUN_TS}.txt"
  done
}

summary_queues() {
  local launched_file="${1:?launched queue file required}"
  while read -r host remote_rel; do
    [[ -n "${host:-}" && -n "${remote_rel:-}" ]] || continue
    echo "## ${host} ${remote_rel}"
    ssh_cmd "${host}" "python3 - \"\$HOME/${remote_rel}/queue_state.json\" \"\$HOME/${remote_rel}/logs\" <<'PY'
import json, os, re, sys
state_path, log_dir = sys.argv[1], sys.argv[2]
if not os.path.exists(state_path):
    print('queue_state: missing')
    raise SystemExit(0)
with open(state_path) as handle:
    state = json.load(handle)
jobs = state.get('jobs', [])
counts = {}
for job in jobs:
    counts[job.get('status', 'unknown')] = counts.get(job.get('status', 'unknown'), 0) + 1
print('counts:', counts)
for job in [j for j in jobs if j.get('status') == 'running'][:5]:
    print('running:', job.get('id'), 'gpu=', job.get('gpu'), 'attempts=', job.get('attempts'))
problems = [j for j in jobs if j.get('status') in {'stuck', 'failed_oom', 'failed_other'} or j.get('error')]
for job in problems[:8]:
    print('problem:', job.get('id'), job.get('status'), job.get('error'))
pattern = re.compile(r'(Traceback|out of memory|CUDA out of memory|\\bnan\\b)', re.I)
hits = []
if os.path.isdir(log_dir):
    for name in sorted(os.listdir(log_dir))[-40:]:
        if name.endswith('.log'):
            text = open(os.path.join(log_dir, name), errors='replace').read()[-20000:]
            if pattern.search(text):
                hits.append(name)
print('log_alerts:', ','.join(hits[:8]) if hits else 'none')
PY
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -4 || true"
  done < "${launched_file}"
}

fetch_queues() {
  local launched_file="${1:?launched queue file required}"
  local dest_root="${2:?destination directory required}"
  local result_match="${RESULT_MATCH:-}"
  mkdir -p "${dest_root}"
  while read -r host remote_rel; do
    [[ -n "${host:-}" && -n "${remote_rel:-}" ]] || continue
    local host_dir="${dest_root}/${host}"
    mkdir -p "${host_dir}"
    rsync_pull_cmd "${host}" "\$HOME/${remote_rel}/" "${host_dir}/queue/"
    if [[ -n "${result_match}" ]]; then
      rsync_pull_cmd "${host}" "${REMOTE_DIR}/PatchTST/PatchTST_supervised/results/" "${host_dir}/results/" \
        --prune-empty-dirs --include '*/' --include 'summary.csv' \
        --include "*${result_match}*/config.json" --include "*${result_match}*/metrics.npy" \
        --include "*${result_match}*/pred.npy" --include "*${result_match}*/true.npy" --exclude '*'
    else
      rsync_pull_cmd "${host}" "${REMOTE_DIR}/PatchTST/PatchTST_supervised/results/" "${host_dir}/results/" \
        --prune-empty-dirs --include '*/' --include 'summary.csv' --include 'config.json' --include 'metrics.npy' --include 'pred.npy' --include 'true.npy' --exclude '*'
    fi
  done < "${launched_file}"
}

case "${1:-inventory}" in
  inventory) inventory ;;
  sync) sync_all ;;
  launch) shift; launch_queues "$@" ;;
  summary) shift; summary_queues "$@" ;;
  fetch) shift; fetch_queues "$@" ;;
  *)
    echo "Usage: $0 {inventory|sync|launch <manifest-dir>|summary <launched-file>|fetch <launched-file> <dest-root>}" >&2
    exit 2
    ;;
esac
