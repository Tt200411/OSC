#!/usr/bin/env bash
set -euo pipefail

HOSTS_DEFAULT="10.21.53.62 10.21.53.82 10.21.53.113 10.21.53.142 10.21.53.162"
DISABLED_HOSTS="10.20.12.248 10.20.12.247"
REMOTE_USER="${REMOTE_USER:-testsv}"
REMOTE_DIR="${REMOTE_DIR:-~/project/osc_informer}"
HOSTS="${HOSTS:-${HOSTS_DEFAULT}}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-.aris/phase4_remote_${RUN_TS}}"
QUEUE_MANAGER="${QUEUE_MANAGER:-lee_ocil/scripts/phase4_queue_manager.py}"
CONDA_ENV="${CONDA_ENV:-torch}"

mkdir -p "${OUT_DIR}"

ssh_cmd() {
  local host="$1"
  shift
  if [[ -n "${PHASE4_SSH_PASSWORD:-}" ]]; then
    export PHASE4_SSH_PASSWORD
    expect -f - "${REMOTE_USER}" "${host}" "$@" <<'EOF'
set timeout -1
set user [lindex $argv 0]
set host [lindex $argv 1]
set password $env(PHASE4_SSH_PASSWORD)
set args [lrange $argv 2 end]
log_user 0
spawn ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 "$user@$host" {*}$args
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
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${REMOTE_USER}@${host}" "$@"
  fi
}

scp_cmd() {
  local source="$1"
  local host="$2"
  local dest="$3"
  if [[ -n "${PHASE4_SSH_PASSWORD:-}" ]]; then
    export PHASE4_SSH_PASSWORD
    expect -f - "${source}" "${REMOTE_USER}" "${host}" "${dest}" <<'EOF'
set timeout -1
set source [lindex $argv 0]
set user [lindex $argv 1]
set host [lindex $argv 2]
set dest [lindex $argv 3]
set password $env(PHASE4_SSH_PASSWORD)
log_user 0
spawn scp -o StrictHostKeyChecking=accept-new "$source" "$user@$host:$dest"
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
    scp "${source}" "${REMOTE_USER}@${host}:${dest}"
  fi
}

rsync_pull_cmd() {
  local host="$1"
  local source="$2"
  local dest="$3"
  shift 3
  local -a rsync_opts
  rsync_opts=("$@")
  mkdir -p "${dest}"
  if [[ -n "${PHASE4_SSH_PASSWORD:-}" ]]; then
    export PHASE4_SSH_PASSWORD
    if (( ${#rsync_opts[@]} )); then
      expect -f - "${REMOTE_USER}" "${host}" "${source}" "${dest}" "${rsync_opts[@]}" <<'EOF'
set timeout -1
set user [lindex $argv 0]
set host [lindex $argv 1]
set source [lindex $argv 2]
set dest [lindex $argv 3]
set rsync_opts [lrange $argv 4 end]
set password $env(PHASE4_SSH_PASSWORD)
log_user 0
spawn rsync -av {*}$rsync_opts -e "ssh -o StrictHostKeyChecking=accept-new" "$user@$host:$source" "$dest"
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
      expect -f - "${REMOTE_USER}" "${host}" "${source}" "${dest}" <<'EOF'
set timeout -1
set user [lindex $argv 0]
set host [lindex $argv 1]
set source [lindex $argv 2]
set dest [lindex $argv 3]
set rsync_opts {}
set password $env(PHASE4_SSH_PASSWORD)
log_user 0
spawn rsync -av {*}$rsync_opts -e "ssh -o StrictHostKeyChecking=accept-new" "$user@$host:$source" "$dest"
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
    fi
  else
    if (( ${#rsync_opts[@]} )); then
      rsync -av "${rsync_opts[@]}" -e "ssh -o BatchMode=yes" "${REMOTE_USER}@${host}:${source}" "${dest}"
    else
      rsync -av -e "ssh -o BatchMode=yes" "${REMOTE_USER}@${host}:${source}" "${dest}"
    fi
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
    output="$(printf '%s\n' "${output}" | tr -d '\r' | sed '/^[[:space:]]*$/d')"
    gpu_line="$(printf '%s\n' "${output}" | sed -n '1p')"
    disk_line="$(printf '%s\n' "${output}" | sed -n '2p')"
    data_count="$(printf '%s\n' "${output}" | sed -n '3p' | tr -d '\r')"
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
    scp_cmd "${QUEUE_MANAGER}" "${host}" ".aris_queue/phase4_queue_manager.py"
    scp_cmd "${manifest}" "${host}" "${remote_rel}/manifest.json"
    screen_name="P4QM_${RUN_TS}_${host//./_}"
    ssh_cmd "${host}" "screen -S ${screen_name} -X quit >/dev/null 2>&1 || true; screen -dmS ${screen_name} bash -lc 'cd ${REMOTE_DIR}/lee_ocil && .venv/bin/python \"\$HOME/.aris_queue/phase4_queue_manager.py\" --manifest \"\$HOME/${remote_rel}/manifest.json\" --state \"\$HOME/${remote_rel}/queue_state.json\" --log-dir \"\$HOME/${remote_rel}/logs\" > \"\$HOME/${remote_rel}/queue_mgr.log\" 2>&1'"
    echo "${host} ${remote_rel}" >> "${OUT_DIR}/phase4_launched_queues_${RUN_TS}.txt"
  done
}

queue_state() {
  local launched_file="${1:?launched queue file required}"
  while read -r host remote_rel; do
    [[ -n "${host:-}" && -n "${remote_rel:-}" ]] || continue
    echo "## ${host} ${remote_rel}"
    ssh_cmd "${host}" "python3 - \"\$HOME/${remote_rel}/queue_state.json\" <<'PY'
import json, os, sys
path = sys.argv[1]
if not os.path.exists(path):
    print('queue_state: missing')
    raise SystemExit(0)
with open(path) as handle:
    state = json.load(handle)
jobs = state.get('jobs', [])
counts = {}
for job in jobs:
    counts[job.get('status', 'unknown')] = counts.get(job.get('status', 'unknown'), 0) + 1
print('counts:', counts)
for job in jobs[:8]:
    print(job.get('id'), job.get('status'), 'gpu=', job.get('gpu'), 'attempts=', job.get('attempts'), 'error=', job.get('error'))
PY
echo '-- queue_mgr.log --'
tail -20 \"\$HOME/${remote_rel}/queue_mgr.log\" 2>/dev/null || true
echo '-- recent job logs --'
ls -t \"\$HOME/${remote_rel}/logs\"/*.log 2>/dev/null | head -3 | while read f; do
  echo \"### \$(basename \"\$f\")\"
  tr '\r' '\n' < \"\$f\" | grep -E 'Use GPU|开始训练|Epoch:|Train Loss|mse:|mae:|Traceback|out of memory|nan|inf' | tail -20 || true
done"
  done < "${launched_file}"
}

queue_summary() {
  local launched_file="${1:?launched queue file required}"
  while read -r host remote_rel; do
    [[ -n "${host:-}" && -n "${remote_rel:-}" ]] || continue
    echo "## ${host} ${remote_rel}"
    ssh_cmd "${host}" "python3 - \"\$HOME/${remote_rel}/queue_state.json\" \"\$HOME/${remote_rel}/logs\" <<'PY'
import json
import os
import re
import sys

state_path, log_dir = sys.argv[1], sys.argv[2]
if not os.path.exists(state_path):
    print('queue_state: missing')
    raise SystemExit(0)

with open(state_path, encoding='utf-8') as handle:
    state = json.load(handle)
jobs = state.get('jobs', [])
counts = {}
for job in jobs:
    counts[job.get('status', 'unknown')] = counts.get(job.get('status', 'unknown'), 0) + 1
print('counts:', counts)

running = [job for job in jobs if job.get('status') == 'running']
if running:
    for job in running[:3]:
        print('running:', job.get('id'), 'gpu=', job.get('gpu'), 'attempts=', job.get('attempts'))
else:
    print('running: none')

problem_jobs = [
    job for job in jobs
    if job.get('status') in {'stuck', 'failed_oom', 'failed_other'} or job.get('error')
]
if problem_jobs:
    for job in problem_jobs[:8]:
        print('problem:', job.get('id'), job.get('status'), job.get('error'))
else:
    print('problems: none')

pattern = re.compile(r'(Traceback|out of memory|CUDA out of memory|\\bnan\\b)', re.I)
hits = []
if os.path.isdir(log_dir):
    for name in sorted(os.listdir(log_dir))[-40:]:
        if not name.endswith('.log'):
            continue
        path = os.path.join(log_dir, name)
        try:
            with open(path, errors='replace') as handle:
                tail = handle.read()[-20000:]
        except OSError:
            continue
        if pattern.search(tail):
            hits.append(name)
if hits:
    print('log_alerts:', ','.join(hits[:8]))
else:
    print('log_alerts: none')
PY
echo '-- gpu --'
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -4 || true
echo '-- manager screen --'
screen -ls 2>/dev/null | grep 'P4QM' || true"
  done < "${launched_file}"
}

fetch_queues() {
  local launched_file="${1:?launched queue file required}"
  local dest_root="${2:?destination directory required}"
  mkdir -p "${dest_root}"
  while read -r host remote_rel; do
    [[ -n "${host:-}" && -n "${remote_rel:-}" ]] || continue
    host_dir="${dest_root}/${host}"
    mkdir -p "${host_dir}"
    rsync_pull_cmd "${host}" "\$HOME/${remote_rel}/" "${host_dir}/queue/"
    rsync_pull_cmd "${host}" "${REMOTE_DIR}/lee_ocil/results/summary.csv" "${host_dir}/results/"
    if [[ -n "${PHASE4_FETCH_RESULT_PATTERN:-}" ]]; then
      rsync_pull_cmd "${host}" "${REMOTE_DIR}/lee_ocil/results/" "${host_dir}/results/" \
        --prune-empty-dirs \
        --include '*/' \
        --include 'summary.csv' \
        --include "${PHASE4_FETCH_RESULT_PATTERN}/***" \
        --exclude '*'
    elif [[ "${PHASE4_FETCH_ARRAYS:-0}" =~ ^(1|true|yes)$ ]]; then
      rsync_pull_cmd "${host}" "${REMOTE_DIR}/lee_ocil/results/" "${host_dir}/results/"
    else
      rsync_pull_cmd "${host}" "${REMOTE_DIR}/lee_ocil/results/" "${host_dir}/results/" \
        --prune-empty-dirs \
        --include '*/' \
        --include 'config.json' \
        --include 'metrics.npy' \
        --exclude '*'
    fi
  done < "${launched_file}"
}

runtime_check() {
  for host in ${HOSTS}; do
    case " ${DISABLED_HOSTS} " in
      *" ${host} "*) continue ;;
    esac
    echo "## ${host}"
    ssh_cmd "${host}" "cd ${REMOTE_DIR}/lee_ocil && \
      echo pwd:\$PWD && \
      command -v screen && screen --version | head -1 && \
      nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits | head -1 && \
      echo conda_path:\$(command -v conda || true) && \
      for hook in \$HOME/anaconda3/etc/profile.d/conda.sh \$HOME/miniconda3/etc/profile.d/conda.sh \$HOME/miniforge3/etc/profile.d/conda.sh /opt/anaconda3/etc/profile.d/conda.sh /opt/miniconda3/etc/profile.d/conda.sh; do [ -f \"\$hook\" ] && echo conda_hook:\$hook; done && \
      for hook in \$HOME/anaconda3/etc/profile.d/conda.sh \$HOME/miniconda3/etc/profile.d/conda.sh \$HOME/miniforge3/etc/profile.d/conda.sh /opt/anaconda3/etc/profile.d/conda.sh /opt/miniconda3/etc/profile.d/conda.sh; do [ -f \"\$hook\" ] && . \"\$hook\" && break; done; \
      conda env list 2>/dev/null || true; \
      echo system_python:\$(command -v python3 || true); \
      python3 - <<'PY'
import importlib.util, sys
print('system_python_version', sys.version.split()[0])
for name in ['torch', 'numpy', 'pandas', 'sklearn']:
    print('system_import', name, bool(importlib.util.find_spec(name)))
PY
      for venv in .venv venv ../.venv ../venv \$HOME/.venv \$HOME/venv \$HOME/project/osc_informer/.venv \$HOME/project/osc_informer/venv \$HOME/project/osc_informer/lee_ocil/.venv \$HOME/project/osc_informer/lee_ocil/venv; do \
        if [ -x \"\$venv/bin/python\" ]; then \
          echo venv_python:\$(cd \"\$(dirname \"\$venv\")\" && pwd)/\$(basename \"\$venv\")/bin/python; \
          \"\$venv/bin/python\" - <<'PY'
import importlib.util, sys
print('venv_python_version', sys.version.split()[0])
for name in ['torch', 'numpy', 'pandas', 'sklearn']:
    print('venv_import', name, bool(importlib.util.find_spec(name)))
PY
        fi; \
      done; \
      for env in torch osc_informer base informer; do conda activate \"\$env\" >/dev/null 2>&1 && echo env_ok:\$env && python - <<'PY'
import importlib.util
print('python_ok')
for name in ['torch', 'numpy', 'pandas']:
    print(name, bool(importlib.util.find_spec(name)))
PY
      conda deactivate >/dev/null 2>&1 || true; done; \
      python3 -m py_compile phase4_informer.py scripts/phase4_common.py scripts/phase4_matrix.py scripts/phase4_aggregate.py"
  done
}

case "${1:-inventory}" in
  inventory) inventory ;;
  sync) sync_all ;;
  launch) shift; launch_queues "$@" ;;
  state) shift; queue_state "$@" ;;
  summary) shift; queue_summary "$@" ;;
  fetch) shift; fetch_queues "$@" ;;
  runtime) runtime_check ;;
  *)
    echo "Usage: $0 {inventory|sync|launch <manifest-dir>|state <launched-file>|summary <launched-file>|fetch <launched-file> <dest-root>|runtime}" >&2
    exit 2
    ;;
esac
