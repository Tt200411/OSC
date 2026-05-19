#!/usr/bin/env bash
set -euo pipefail

HOSTS_DEFAULT="10.21.53.62 10.21.53.82 10.21.53.113 10.21.53.142 10.21.53.162"
DISABLED_HOSTS="10.20.12.248 10.20.12.247"
REMOTE_USER="${REMOTE_USER:-testsv}"
REMOTE_DIR="${REMOTE_DIR:-~/project/osc_informer}"
HOSTS="${HOSTS:-${HOSTS_DEFAULT}}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-.aris/phase4_remote_${RUN_TS}}"
QUEUE_TOOL_DIR="${QUEUE_TOOL_DIR:-/Users/tangbao/.codex/Auto-claude-code-research-in-sleep/tools/experiment_queue}"
CONDA_ENV="${CONDA_ENV:-torch}"

mkdir -p "${OUT_DIR}"

ssh_cmd() {
  local host="$1"
  shift
  if [[ -n "${PHASE4_SSH_PASSWORD:-}" ]]; then
    expect <<'EOF' "${REMOTE_USER}" "${host}" "${PHASE4_SSH_PASSWORD}" "$@"
set timeout -1
set user [lindex $argv 0]
set host [lindex $argv 1]
set password [lindex $argv 2]
set args [lrange $argv 3 end]
spawn ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$user@$host" {*}$args
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
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${REMOTE_USER}@${host}" "$@"
  fi
}

scp_cmd() {
  local source="$1"
  local host="$2"
  local dest="$3"
  if [[ -n "${PHASE4_SSH_PASSWORD:-}" ]]; then
    expect <<'EOF' "${source}" "${REMOTE_USER}" "${host}" "${dest}" "${PHASE4_SSH_PASSWORD}"
set timeout -1
set source [lindex $argv 0]
set user [lindex $argv 1]
set host [lindex $argv 2]
set dest [lindex $argv 3]
set password [lindex $argv 4]
spawn scp -o StrictHostKeyChecking=accept-new "$source" "$user@$host:$dest"
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
    scp "${source}" "${REMOTE_USER}@${host}:${dest}"
  fi
}

write_inventory_header() {
  local path="$1"
  {
    echo "# Phase-4 GPU Inventory"
    echo
    echo "Generated: $(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo
    echo "Disabled hosts: ${DISABLED_HOSTS}"
    echo
    echo "| host | status | gpu | memory.used | memory.total | home_disk | data_12_12 | notes |"
    echo "| --- | --- | --- | --- | --- | --- | --- | --- |"
  } > "${path}"
}

sanitize_note() {
  local text="$1"
  text="${text//$'\r'/ }"
  text="${text//$'\n'/ }"
  if [[ "${text}" == *"Permission denied"* ]]; then
    printf 'Permission denied by SSH authentication; no transient password supplied or accepted.'
  else
    printf '%s' "${text}"
  fi
}

inventory() {
  local path="${OUT_DIR}/phase4_gpu_inventory_${RUN_TS}.md"
  write_inventory_header "${path}"
  for host in ${HOSTS}; do
    case " ${DISABLED_HOSTS} " in
      *" ${host} "*) continue ;;
    esac
    set +e
    output="$(ssh_cmd "${host}" "cd ${REMOTE_DIR} && \
      gpu=\$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || true); \
      disk=\$(df -h \$HOME | tail -1 | awk '{print \$4\" free of \"\$2}'); \
      ok=0; for f in lee_ocil/ETT-small/ETTh1.csv lee_ocil/ETT-small/ETTh2.csv lee_ocil/ETT-small/ETTm1.csv lee_ocil/ETT-small/ETTm2.csv Solar/Site_1_50MW.csv Solar/Site_2_130MW.csv Solar/Site_3_30MW.csv Solar/Site_4_130MW.csv Solar/Site_5_110MW.csv Solar/Site_6_35MW.csv Solar/Site_7_30MW.csv Solar/Site_8_30MW.csv; do [ -f \"\$f\" ] && ok=\$((ok+1)); done; \
      printf '%s\n%s\n%s\n' \"\$gpu\" \"\$disk\" \"\$ok\"" 2>&1)"
    rc=$?
    set -e
    if [[ "${rc}" -ne 0 ]]; then
      note="$(sanitize_note "${output}")"
      echo "| ${host} | auth_or_ssh_failed |  |  |  |  | 0/12 | ${note} |" >> "${path}"
      continue
    fi
    gpu_line="$(printf '%s\n' "${output}" | sed -n '1p')"
    disk_line="$(printf '%s\n' "${output}" | sed -n '2p')"
    data_count="$(printf '%s\n' "${output}" | sed -n '3p')"
    IFS=',' read -r gpu_name mem_used mem_total <<< "${gpu_line}"
    echo "| ${host} | ok | ${gpu_name} | ${mem_used} MiB | ${mem_total} MiB | ${disk_line} | ${data_count}/12 |  |" >> "${path}"
  done
  echo "${path}"
}

sync_all() {
  for host in ${HOSTS}; do
    case " ${DISABLED_HOSTS} " in
      *" ${host} "*) continue ;;
    esac
    PHASE4_SSH_PASSWORD="${PHASE4_SSH_PASSWORD:-}" SSH_OPTS="${SSH_OPTS:-}" \
      scripts/sync_phase4_remote.sh "${REMOTE_USER}@${host}" "${REMOTE_DIR}"
  done
}

launch_queues() {
  local manifest_dir="${1:?manifest dir required}"
  for host in ${HOSTS}; do
    case " ${DISABLED_HOSTS} " in
      *" ${host} "*) continue ;;
    esac
    manifest="${manifest_dir}/phase4_queue_manifest_${host//./_}.json"
    [[ -f "${manifest}" ]] || continue
    remote_rel=".aris_queue/runs/${RUN_TS}_${host//./_}"
    ssh_cmd "${host}" "mkdir -p \"\$HOME/${remote_rel}/logs\" \"\$HOME/.aris_queue\""
    scp_cmd "${QUEUE_TOOL_DIR}/queue_manager.py" "${host}" ".aris_queue/queue_manager.py"
    scp_cmd "${manifest}" "${host}" "${remote_rel}/manifest.json"
    ssh_cmd "${host}" "cd ${REMOTE_DIR} && nohup python3 \"\$HOME/.aris_queue/queue_manager.py\" --manifest \"\$HOME/${remote_rel}/manifest.json\" --state \"\$HOME/${remote_rel}/queue_state.json\" --log-dir \"\$HOME/${remote_rel}/logs\" > \"\$HOME/${remote_rel}/queue_mgr.log\" 2>&1 &"
    echo "${host} ${remote_rel}" >> "${OUT_DIR}/phase4_launched_queues_${RUN_TS}.txt"
  done
}

case "${1:-inventory}" in
  inventory) inventory ;;
  sync) sync_all ;;
  launch) shift; launch_queues "$@" ;;
  *)
    echo "Usage: $0 {inventory|sync|launch <manifest-dir>}" >&2
    exit 2
    ;;
esac
