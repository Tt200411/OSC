#!/usr/bin/env python3
"""Phase-4 remote queue manager.

This is a small, versioned scheduler for the Phase-4 Informer grid.  It runs on
the remote GPU host, launches manifest jobs through screen, writes a durable
queue_state.json, retries CUDA OOM once/twice per manifest policy, and supports
the existing remote venv used by the earlier experiments.
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path


OOM_RE = re.compile(r"(CUDA out of memory|torch\.OutOfMemoryError)", re.I)
FAIL_RE = re.compile(r"(Traceback|ModuleNotFoundError|ImportError|No such file|command not found|CondaError)", re.I)
DEFAULT_GPU_FREE_THRESHOLD_MIB = 500
POLL_INTERVAL_SEC = 60


def now():
    return datetime.utcnow().isoformat() + "Z"


def run(cmd, check=False):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return result.stdout, result.returncode


def gpu_memory_used():
    out, rc = run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
    if rc != 0:
        return []
    return [int(item.strip()) for item in out.splitlines() if item.strip()]


def free_gpus(allowed, threshold_mib):
    used = gpu_memory_used()
    return [idx for idx in allowed if idx < len(used) and used[idx] < threshold_mib]


def screen_exists(name):
    out, _ = run("screen -ls")
    return f".{name}" in out or f"\t{name}" in out


def kill_screen(name):
    run(f"screen -S {shlex.quote(name)} -X quit")


def output_exists(path_pattern, cwd):
    if not path_pattern:
        return False
    full = path_pattern if os.path.isabs(path_pattern) else os.path.join(cwd, path_pattern)
    out, _ = run(f"ls {shlex.quote(full)} 2>/dev/null | wc -l")
    try:
        return int(out.strip()) > 0
    except ValueError:
        return False


def tail_log(path, bytes_=20000):
    try:
        return Path(path).read_text(errors="replace")[-bytes_:]
    except FileNotFoundError:
        return ""


def detect_oom(log_path):
    return bool(OOM_RE.search(tail_log(log_path)))


def detect_failure(log_path):
    text = tail_log(log_path)
    return bool(text and FAIL_RE.search(text))


def python_prefix(manifest):
    """Return shell statements that prepare the requested Python runtime."""
    venv_path = manifest.get("venv_path")
    python = manifest.get("python")
    conda_env = manifest.get("conda")
    conda_hook = manifest.get("conda_hook")

    if venv_path:
        venv_bin = os.path.join(str(venv_path), "bin")
        return (
            f"export VIRTUAL_ENV={shlex.quote(str(venv_path))} && "
            f"export PATH={shlex.quote(venv_bin)}:$PATH"
        )
    if python:
        return "true"
    if conda_env and str(conda_env).lower() not in {"none", "false", "0", "system"}:
        if conda_hook:
            hook = conda_hook if conda_hook.startswith("eval") else f'eval "$({conda_hook} shell.bash hook)"'
        else:
            hook = 'eval "$(conda shell.bash hook)"'
        return f"{hook} && conda activate {shlex.quote(str(conda_env))}"
    return "true"


def python_executable(manifest):
    if manifest.get("python"):
        return str(manifest["python"])
    if manifest.get("venv_path"):
        return os.path.join(str(manifest["venv_path"]), "bin", "python")
    return "python"


def normalize_cmd(job_cmd, manifest):
    py = python_executable(manifest)
    if job_cmd.startswith("python "):
        return py + job_cmd[len("python") :]
    if job_cmd.startswith("python3 "):
        return py + job_cmd[len("python3") :]
    return job_cmd


def load_state(state_file, manifest):
    if Path(state_file).exists():
        with open(state_file, encoding="utf-8") as handle:
            return json.load(handle)
    return {
        "meta": {
            "project": manifest.get("project", "unknown"),
            "started": now(),
            "manifest_path": manifest.get("_path", ""),
        },
        "phases": [
            {
                "name": phase.get("name", f"phase_{idx}"),
                "depends_on": phase.get("depends_on", []),
                "status": "pending",
            }
            for idx, phase in enumerate(manifest.get("phases", []))
        ],
        "jobs": [],
    }


def save_state(state, state_file):
    tmp = state_file + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)
    os.replace(tmp, state_file)


def assign_jobs(manifest, state):
    for phase in manifest.get("phases", []):
        phase_name = phase.get("name")
        for job in phase.get("jobs", []):
            if any(existing["id"] == job["id"] for existing in state["jobs"]):
                continue
            state["jobs"].append(
                {
                    "id": job["id"],
                    "phase": phase_name,
                    "cmd": job["cmd"],
                    "expected_output": job.get("expected_output"),
                    "status": "pending",
                    "gpu": None,
                    "screen_name": None,
                    "pid": None,
                    "attempts": 0,
                    "started": None,
                    "completed": None,
                    "error": None,
                }
            )


def phase_ready(phase_name, state):
    phase = next((item for item in state["phases"] if item["name"] == phase_name), None)
    if not phase:
        return False
    for dep in phase.get("depends_on", []):
        dep_phase = next((item for item in state["phases"] if item["name"] == dep), None)
        if not dep_phase or dep_phase.get("status") != "completed":
            return False
    return True


def phase_complete(phase_name, state):
    jobs = [job for job in state["jobs"] if job.get("phase") == phase_name]
    return bool(jobs) and all(job["status"] in {"completed", "stuck"} for job in jobs)


def active_pending_jobs(manifest, state):
    active = {
        phase.get("name")
        for phase in manifest.get("phases", [])
        if phase_ready(phase.get("name"), state) and not phase_complete(phase.get("name"), state)
    }
    return [job for job in state["jobs"] if job["status"] == "pending" and job.get("phase") in active]


def launch_job(job, gpu, manifest, log_dir):
    screen_name = f"EQ_{job['id']}"
    if screen_exists(screen_name):
        kill_screen(screen_name)
        time.sleep(2)

    cwd = manifest.get("cwd", ".")
    log_file = os.path.join(log_dir, f"{job['id']}.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    cmd = normalize_cmd(job["cmd"].replace("${GPU}", str(gpu)), manifest)
    runtime = python_prefix(manifest)
    full = (
        "set -e; "
        f"cd {shlex.quote(cwd)}; "
        f"{runtime}; "
        f"export CUDA_VISIBLE_DEVICES={gpu}; "
        f"{cmd}"
    )
    logged = "{ " + full + "; } 2>&1 | tee " + shlex.quote(log_file)
    screen_cmd = f"screen -dmS {shlex.quote(screen_name)} bash -lc {shlex.quote(logged)}"
    run(screen_cmd)
    time.sleep(2)
    pid_out, _ = run(
        f"ps -ef | grep {shlex.quote(job['id'])} | grep -v grep | "
        "grep -E 'python|phase4_informer' | awk '{print $2}' | head -1"
    )
    pid = pid_out.strip()
    return screen_name, (int(pid) if pid.isdigit() else None)


def job_status(job, manifest, log_dir):
    cwd = manifest.get("cwd", ".")
    log_file = os.path.join(log_dir, f"{job['id']}.log")
    if output_exists(job.get("expected_output"), cwd):
        return "completed", None
    if detect_oom(log_file):
        return "failed_oom", "CUDA OOM detected"
    if job.get("screen_name") and screen_exists(job["screen_name"]):
        return "running", None
    if detect_failure(log_file):
        return "failed_other", "Training command failed; inspect job log"
    return "failed_other", "Screen exited without expected output"


def step(manifest, state, state_file, log_dir):
    allowed = manifest.get("gpus", [0])
    max_parallel = manifest.get("max_parallel", len(allowed))
    threshold = manifest.get("gpu_free_threshold_mib", DEFAULT_GPU_FREE_THRESHOLD_MIB)
    oom_delay = manifest.get("oom_retry", {}).get("delay", 180)
    max_attempts = manifest.get("oom_retry", {}).get("max_attempts", 2)

    for job in state["jobs"]:
        if job["status"] != "running":
            continue
        new_status, error = job_status(job, manifest, log_dir)
        if new_status == "running":
            continue
        job["status"] = "stuck" if new_status == "failed_other" else new_status
        job["error"] = error
        job["completed"] = now()
        if job.get("screen_name"):
            kill_screen(job["screen_name"])

    for job in state["jobs"]:
        if job["status"] != "failed_oom":
            continue
        if job["attempts"] >= max_attempts:
            job["status"] = "stuck"
            continue
        if job.get("completed"):
            last = datetime.fromisoformat(job["completed"].rstrip("Z"))
            if (datetime.utcnow() - last).total_seconds() >= oom_delay:
                job["status"] = "pending"
                job["error"] = None

    running = [job for job in state["jobs"] if job["status"] == "running"]
    pending = active_pending_jobs(manifest, state)
    taken = {job["gpu"] for job in running if job.get("gpu") is not None}
    free = [gpu for gpu in free_gpus(allowed, threshold) if gpu not in taken]
    slots = min(max_parallel - len(running), len(free), len(pending))
    for idx in range(slots):
        job = pending[idx]
        screen_name, pid = launch_job(job, free[idx], manifest, log_dir)
        job["status"] = "running"
        job["gpu"] = free[idx]
        job["screen_name"] = screen_name
        job["pid"] = pid
        job["attempts"] += 1
        job["started"] = now()
        job["completed"] = None
        job["error"] = None

    for phase in state["phases"]:
        if phase_complete(phase["name"], state):
            phase["status"] = "completed"
        elif any(job["status"] == "running" for job in state["jobs"] if job.get("phase") == phase["name"]):
            phase["status"] = "running"

    save_state(state, state_file)


def all_done(state):
    return all(job["status"] in {"completed", "stuck"} for job in state["jobs"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--poll", type=int, default=POLL_INTERVAL_SEC)
    args = parser.parse_args()

    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["_path"] = args.manifest
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    state = load_state(args.state, manifest)
    assign_jobs(manifest, state)
    save_state(state, args.state)
    print(f"[{now()}] Phase-4 queue manager started with {len(state['jobs'])} jobs", flush=True)

    while not all_done(state):
        try:
            step(manifest, state, args.state, args.log_dir)
        except Exception as exc:
            print(f"[{now()}] Step error: {exc}", flush=True)
        time.sleep(args.poll)

    print(f"[{now()}] All jobs done", flush=True)


if __name__ == "__main__":
    main()
