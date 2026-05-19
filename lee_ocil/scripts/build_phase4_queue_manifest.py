#!/usr/bin/env python3
import argparse
import json
import os
import shlex
from datetime import datetime

import pandas as pd

from phase4_common import CONFIG_FLAGS, root_path_for_dataset, target_for_dataset, freq_for_dataset


HOST_PRIORITY = ["10.21.53.62", "10.21.53.82", "10.21.53.113", "10.21.53.142", "10.21.53.162"]
SMOKE_ROWS = {
    ("ETTm1", 96, "gelu_all", 2024),
    ("ETTm2", 96, "gelu_all", 2024),
    ("Solar2", 96, "gelu_all", 2024),
    ("Solar8", 96, "gelu_all", 2024),
}


def job_priority(row):
    dataset = row["dataset"]
    pred_len = int(row["pred_len"])
    if dataset.startswith("ETT"):
        group = 0
    elif pred_len == 96:
        group = 1
    else:
        group = 2
    return (group, dataset, pred_len, row["config_name"], int(row["seed"]))


def row_is_smoke(row):
    return (
        row["dataset"],
        int(row["pred_len"]),
        row["config_name"],
        int(row["seed"]),
    ) in SMOKE_ROWS


def shell_join(args):
    return " ".join(shlex.quote(str(arg)) for arg in args)


def server_id_for_host(host):
    return host.replace("10.21.53.", "4090-")


def command_for_row(row, run_tag, save_arrays, server_id, server_ip):
    config = CONFIG_FLAGS[row["config_name"]]
    args = [
        "python",
        "-u",
        "phase4_informer.py",
        "--model",
        "informer",
        "--features",
        "M",
        "--d_model",
        512,
        "--n_heads",
        8,
        "--d_ff",
        2048,
        "--attn",
        "prob",
        "--embed",
        "timeF",
        "--distil",
        "true",
        "--mix",
        "true",
        "--batch_size",
        int(row["batch_size"]),
        "--learning_rate",
        "1e-4",
        "--patience",
        3,
        "--lradj",
        "type1",
        "--train_epochs",
        int(row["train_epochs"]),
        "--dropout",
        0.05,
        "--data",
        row["dataset"],
        "--root_path",
        root_path_for_dataset(row["dataset"]),
        "--data_path",
        row["data_path"],
        "--target",
        target_for_dataset(row["dataset"]),
        "--freq",
        freq_for_dataset(row["dataset"]),
        "--detail_freq",
        freq_for_dataset(row["dataset"]),
        "--seq_len",
        int(row["seq_len"]),
        "--label_len",
        int(row["label_len"]),
        "--pred_len",
        int(row["pred_len"]),
        "--e_layers",
        int(row["e_layers"]),
        "--d_layers",
        int(row["d_layers"]),
        "--factor",
        int(row["factor"]),
        "--activation",
        config["activation"],
        "--encoder_activation",
        config["encoder_activation"],
        "--decoder_activation",
        config["decoder_activation"],
        "--output_activation",
        config["output_activation"],
        "--perturb_amplitude",
        config.get("amplitude", 0.0),
        "--perturb_frequency",
        config.get("frequency", 1.0),
        "--perturb_phase",
        config.get("phase", 0.0),
        "--lee_type",
        config.get("lee_type", 1),
        "--seed",
        int(row["seed"]),
        "--gpu",
        "${GPU}",
        "--server_id",
        server_id,
        "--server_ip",
        server_ip,
        "--run_id",
        run_tag,
        "--des",
        f"{run_tag}_{row['config_name']}",
    ]
    if not save_arrays:
        args.extend(["--no_save_pred", "--no_save_true"])
    return shell_join(args)


def expected_output_for_row(row, run_tag):
    config = CONFIG_FLAGS[row["config_name"]]
    parts = [
        "informer",
        row["dataset"],
        config["activation"],
        f"a{float(config.get('amplitude', 0.0)):g}",
    ]
    if config["activation"] == "lee":
        parts.append(f"lee{int(config.get('lee_type', 1))}")
    if (
        config["encoder_activation"] != config["activation"]
        or config["decoder_activation"] != config["activation"]
        or config["output_activation"] != "linear"
    ):
        parts.append(
            f"enc{config['encoder_activation']}_dec{config['decoder_activation']}_out{config['output_activation']}"
        )
    parts.extend(
        [
            "ftM",
            f"sl{int(row['seq_len'])}",
            f"ll{int(row['label_len'])}",
            f"pl{int(row['pred_len'])}",
            "dm512",
            "nh8",
            f"el{int(row['e_layers'])}",
            f"dl{int(row['d_layers'])}",
            "df2048",
            f"fc{int(row['factor'])}",
            f"seed{int(row['seed'])}",
            "SERVER_ID",
            f"{run_tag}_{row['config_name']}",
        ]
    )
    return "results/" + "_".join(parts) + "/config.json"


def split_host(row, hosts):
    dataset = row["dataset"]
    if dataset.startswith("ETTh") or dataset.startswith("ETTm"):
        idx = (["ETTh1", "ETTh2", "ETTm1", "ETTm2"].index(dataset) + int(row["pred_len"])) % len(hosts)
        return hosts[idx]
    solar_idx = int(dataset.replace("Solar", "")) - 1
    return hosts[solar_idx % len(hosts)]


def build_manifests(
    missing,
    hosts,
    run_tag,
    remote_cwd,
    conda_env,
    venv_path,
    python,
    max_parallel,
    gpus,
    save_arrays,
):
    manifests = {host: [] for host in hosts}
    rows = sorted(missing.to_dict("records"), key=job_priority)
    for row in rows:
        host = split_host(row, hosts)
        server_id = server_id_for_host(host)
        job_id = (
            f"p4_{row['dataset']}_pl{int(row['pred_len'])}_{row['config_name']}_s{int(row['seed'])}"
        )
        expected_output = expected_output_for_row(row, run_tag).replace("SERVER_ID", server_id)
        manifests[host].append(
            {
                "id": job_id,
                "cmd": command_for_row(row, run_tag, save_arrays, server_id=server_id, server_ip=host),
                "expected_output": expected_output,
            }
        )

    payloads = {}
    for host, jobs in manifests.items():
        phases = []
        assigned = {name: [] for name in ["smoke", "ett_ettm", "solar96", "solar24", "other"]}
        phase_specs = [
            ("smoke", lambda job: any(token in job["id"] for token in ["ETTm1_pl96_gelu_all_s2024", "ETTm2_pl96_gelu_all_s2024", "Solar2_pl96_gelu_all_s2024", "Solar8_pl96_gelu_all_s2024"])),
            ("ett_ettm", lambda job: "_ETT" in job["id"]),
            ("solar96", lambda job: "_Solar" in job["id"] and "_pl96_" in job["id"]),
            ("solar24", lambda job: "_Solar" in job["id"] and "_pl24_" in job["id"]),
        ]
        for job in jobs:
            for phase_name, predicate in phase_specs:
                if predicate(job):
                    assigned[phase_name].append(job)
                    break
            else:
                assigned["other"].append(job)
        has_smoke = bool(assigned["smoke"])
        for phase_name, predicate in [
            ("smoke", None),
            ("ett_ettm", None),
            ("solar96", None),
            ("solar24", None),
            ("other", None),
        ]:
            phase_jobs = assigned[phase_name]
            if phase_jobs:
                depends_on = ["smoke"] if phase_name != "smoke" and has_smoke else []
                phases.append({"name": phase_name, "depends_on": depends_on, "jobs": phase_jobs})
        payloads[host] = {
            "project": f"phase4_full_grid_{host}",
            "cwd": remote_cwd,
            "conda": conda_env,
            "venv_path": venv_path,
            "python": python,
            "gpus": gpus,
            "max_parallel": max_parallel,
            "gpu_free_threshold_mib": 500,
            "oom_retry": {"delay": 180, "max_attempts": 2},
            "phases": phases,
        }
    return payloads


def main():
    parser = argparse.ArgumentParser(description="Build per-host ARIS queue manifests from Phase-4 missing runs")
    parser.add_argument("missing_runs_csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--run_tag", default="")
    parser.add_argument("--host", action="append", default=[])
    parser.add_argument("--remote_cwd", default="/home/testsv/project/osc_informer/lee_ocil")
    parser.add_argument("--conda_env", default="none")
    parser.add_argument("--venv_path", default="/home/testsv/project/osc_informer/lee_ocil/.venv")
    parser.add_argument("--python", default="")
    parser.add_argument("--max_parallel", type=int, default=1)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--save_arrays", action="store_true")
    parser.add_argument("--only_smoke", action="store_true")
    parser.add_argument("--exclude_smoke", action="store_true")
    args = parser.parse_args()
    if args.only_smoke and args.exclude_smoke:
        raise SystemExit("--only_smoke and --exclude_smoke are mutually exclusive")

    os.makedirs(args.output_dir, exist_ok=True)
    missing = pd.read_csv(args.missing_runs_csv)
    if args.only_smoke:
        missing = missing[missing.apply(row_is_smoke, axis=1)].copy()
    if args.exclude_smoke:
        missing = missing[~missing.apply(row_is_smoke, axis=1)].copy()
    run_tag = args.run_tag or f"phase4_full_grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    hosts = args.host or HOST_PRIORITY
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    payloads = build_manifests(
        missing,
        hosts=hosts,
        run_tag=run_tag,
        remote_cwd=args.remote_cwd,
        conda_env=args.conda_env,
        venv_path=args.venv_path,
        python=args.python,
        max_parallel=args.max_parallel,
        gpus=gpus,
        save_arrays=args.save_arrays,
    )

    summary_rows = []
    for host, payload in payloads.items():
        job_count = sum(len(phase["jobs"]) for phase in payload["phases"])
        if job_count == 0:
            continue
        path = os.path.join(args.output_dir, f"phase4_queue_manifest_{host.replace('.', '_')}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        summary_rows.append({"host": host, "jobs": job_count, "manifest": path})
        print(f"Wrote {path}: {job_count} jobs")
    pd.DataFrame(summary_rows).to_csv(os.path.join(args.output_dir, "phase4_queue_manifest_summary.csv"), index=False)


if __name__ == "__main__":
    main()
