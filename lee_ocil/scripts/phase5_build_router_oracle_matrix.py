#!/usr/bin/env python3
import argparse
import os
import shlex

import pandas as pd

from phase4_common import CANONICAL_PROTOCOL, CONFIG_FLAGS


DATASETS = {
    "Weather": {
        "root_path": "../Weather",
        "data_path": "weather.csv",
        "target": "OT",
        "freq": "t",
        "horizons": [96, 192, 336],
        "seq_len": 96,
        "label_len": 48,
        "e_layers": 2,
        "d_layers": 1,
        "factor": 5,
    },
    "Exchange": {
        "root_path": "../Exchange",
        "data_path": "exchange_rate.csv",
        "target": "OT",
        "freq": "d",
        "horizons": [96, 192, 336],
        "seq_len": 96,
        "label_len": 48,
        "e_layers": 2,
        "d_layers": 1,
        "factor": 5,
    },
    "ILI": {
        "root_path": "../ILI",
        "data_path": "national_illness.csv",
        "target": "OT",
        "freq": "w",
        "horizons": [24, 36, 48],
        "seq_len": 36,
        "label_len": 18,
        "e_layers": 2,
        "d_layers": 1,
        "factor": 5,
    },
}

ORACLE_CONFIGS = [
    "gelu_all",
    "tanh_sin001_all",
    "tanh_all",
    "softsign_all",
    "enc_tanh_dec_gelu",
    "enc_gelu_dec_tanh",
    "tanh_all_outtanh",
]


def shell_join(items):
    return " ".join(shlex.quote(str(item)) for item in items)


def command_for(row):
    cfg = CONFIG_FLAGS[row["config_name"]]
    args = [
        "python",
        "-u",
        "phase4_informer.py",
        "--model",
        "informer",
        "--features",
        "M",
        "--d_model",
        CANONICAL_PROTOCOL["d_model"],
        "--n_heads",
        CANONICAL_PROTOCOL["n_heads"],
        "--d_ff",
        CANONICAL_PROTOCOL["d_ff"],
        "--attn",
        CANONICAL_PROTOCOL["attn"],
        "--embed",
        CANONICAL_PROTOCOL["embed"],
        "--distil",
        "true",
        "--mix",
        "true",
        "--batch_size",
        CANONICAL_PROTOCOL["batch_size"],
        "--learning_rate",
        "1e-4",
        "--patience",
        3,
        "--lradj",
        "type1",
        "--train_epochs",
        CANONICAL_PROTOCOL["train_epochs"],
        "--dropout",
        CANONICAL_PROTOCOL["dropout"],
        "--data",
        row["dataset"],
        "--root_path",
        row["root_path"],
        "--data_path",
        row["data_path"],
        "--target",
        row["target"],
        "--freq",
        row["freq"],
        "--detail_freq",
        row["freq"],
        "--seq_len",
        row["seq_len"],
        "--label_len",
        row["label_len"],
        "--pred_len",
        row["pred_len"],
        "--e_layers",
        row["e_layers"],
        "--d_layers",
        row["d_layers"],
        "--factor",
        row["factor"],
        "--activation",
        cfg["activation"],
        "--encoder_activation",
        cfg["encoder_activation"],
        "--decoder_activation",
        cfg["decoder_activation"],
        "--output_activation",
        cfg["output_activation"],
        "--perturb_amplitude",
        cfg.get("amplitude", 0.0),
        "--perturb_frequency",
        cfg.get("frequency", 1.0),
        "--perturb_phase",
        cfg.get("phase", 0.0),
        "--lee_type",
        cfg.get("lee_type", 1),
        "--seed",
        row["seed"],
        "--gpu",
        "${GPU}",
        "--server_id",
        "${SERVER_ID}",
        "--server_ip",
        "${SERVER_IP}",
        "--run_id",
        row["run_tag"],
        "--des",
        f"{row['run_tag']}_{row['config_name']}",
    ]
    if row["save_arrays"] is False:
        args.extend(["--no_save_pred", "--no_save_true"])
    return shell_join(args)


def build_matrix(args):
    rows = []
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    datasets = args.datasets or list(DATASETS)
    configs = args.configs or ORACLE_CONFIGS
    for dataset in datasets:
        spec = DATASETS[dataset]
        for pred_len in spec["horizons"]:
            for config_name in configs:
                for seed in seeds:
                    rows.append(
                        {
                            "phase": "oracle_seed1" if seeds == [2024] else "oracle_multiseed",
                            "dataset": dataset,
                            "pred_len": pred_len,
                            "config_name": config_name,
                            "seed": seed,
                            "root_path": spec["root_path"],
                            "data_path": spec["data_path"],
                            "target": spec["target"],
                            "freq": spec["freq"],
                            "seq_len": spec["seq_len"],
                            "label_len": spec["label_len"],
                            "e_layers": spec["e_layers"],
                            "d_layers": spec["d_layers"],
                            "factor": spec["factor"],
                            "batch_size": CANONICAL_PROTOCOL["batch_size"],
                            "train_epochs": CANONICAL_PROTOCOL["train_epochs"],
                            "run_tag": args.run_tag,
                            "save_arrays": bool(args.save_arrays),
                        }
                    )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["cmd"] = frame.apply(command_for, axis=1)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Build Phase-5 Weather/Exchange/ILI oracle matrix")
    parser.add_argument("--output", default=".aris/phase5_new_dataset_oracle_matrix_20260531.csv")
    parser.add_argument("--run_tag", default="phase5_router_oracle_20260531")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=None)
    parser.add_argument("--configs", nargs="+", choices=sorted(CONFIG_FLAGS), default=None)
    parser.add_argument("--seeds", default="2024")
    parser.add_argument("--save_arrays", action="store_true")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    frame = build_matrix(args)
    frame.to_csv(args.output, index=False)
    summary = (
        frame.groupby(["dataset", "pred_len"], dropna=False)
        .size()
        .reset_index(name="runs")
        .to_string(index=False)
        if not frame.empty
        else "No runs."
    )
    print({"output": args.output, "runs": int(len(frame))})
    print(summary)


if __name__ == "__main__":
    main()
