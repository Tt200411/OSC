import glob
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

LEE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if LEE_ROOT not in sys.path:
    sys.path.insert(0, LEE_ROOT)

from models.activations import activation_family


SEEDS = [2024, 2025, 2026]
CONFIG_NAMES = [
    "gelu_all",
    "relu_all",
    "swish_all",
    "tanh_all",
    "softsign_all",
    "tanh_sin001_all",
    "enc_tanh_dec_gelu",
    "enc_gelu_dec_tanh",
    "tanh_all_outtanh",
]

CANONICAL_PROTOCOL = {
    "features": "M",
    "d_model": 512,
    "n_heads": 8,
    "d_ff": 2048,
    "dropout": 0.05,
    "attn": "prob",
    "embed": "timeF",
    "batch_size": 32,
    "train_epochs": 6,
}

ETT_FILES = {
    "ETTh1": "lee_ocil/ETT-small/ETTh1.csv",
    "ETTh2": "lee_ocil/ETT-small/ETTh2.csv",
    "ETTm1": "lee_ocil/ETT-small/ETTm1.csv",
    "ETTm2": "lee_ocil/ETT-small/ETTm2.csv",
}

SOLAR_FILES = {
    "Solar1": "Solar/Site_1_50MW.csv",
    "Solar2": "Solar/Site_2_130MW.csv",
    "Solar3": "Solar/Site_3_30MW.csv",
    "Solar4": "Solar/Site_4_130MW.csv",
    "Solar5": "Solar/Site_5_110MW.csv",
    "Solar6": "Solar/Site_6_35MW.csv",
    "Solar7": "Solar/Site_7_30MW.csv",
    "Solar8": "Solar/Site_8_30MW.csv",
}

DATA_FILES = {**ETT_FILES, **SOLAR_FILES}

REMOTE_DATA_FILES = {
    "ETTh1": "lee_ocil/ETT-small/ETTh1.csv",
    "ETTh2": "lee_ocil/ETT-small/ETTh2.csv",
    "ETTm1": "lee_ocil/ETT-small/ETTm1.csv",
    "ETTm2": "lee_ocil/ETT-small/ETTm2.csv",
    "Solar1": "Solar/Site_1_50MW.csv",
    "Solar2": "Solar/Site_2_130MW.csv",
    "Solar3": "Solar/Site_3_30MW.csv",
    "Solar4": "Solar/Site_4_130MW.csv",
    "Solar5": "Solar/Site_5_110MW.csv",
    "Solar6": "Solar/Site_6_35MW.csv",
    "Solar7": "Solar/Site_7_30MW.csv",
    "Solar8": "Solar/Site_8_30MW.csv",
}

HORIZONS = {
    "ETTh1": {
        24: {"seq_len": 48, "label_len": 48, "e_layers": 2, "d_layers": 1, "factor": 3},
        48: {"seq_len": 96, "label_len": 48, "e_layers": 2, "d_layers": 1, "factor": 5},
        168: {"seq_len": 168, "label_len": 168, "e_layers": 2, "d_layers": 1, "factor": 5},
        336: {"seq_len": 168, "label_len": 168, "e_layers": 2, "d_layers": 1, "factor": 5},
        720: {"seq_len": 336, "label_len": 336, "e_layers": 2, "d_layers": 1, "factor": 5},
    },
    "ETTh2": {
        24: {"seq_len": 48, "label_len": 48, "e_layers": 2, "d_layers": 1, "factor": 5},
        48: {"seq_len": 96, "label_len": 96, "e_layers": 2, "d_layers": 1, "factor": 5},
        168: {"seq_len": 336, "label_len": 336, "e_layers": 3, "d_layers": 2, "factor": 5},
        336: {"seq_len": 336, "label_len": 168, "e_layers": 3, "d_layers": 2, "factor": 5},
        720: {"seq_len": 720, "label_len": 336, "e_layers": 3, "d_layers": 2, "factor": 5},
    },
    "ETTm1": {
        24: {"seq_len": 672, "label_len": 96, "e_layers": 2, "d_layers": 1, "factor": 5},
        48: {"seq_len": 96, "label_len": 48, "e_layers": 2, "d_layers": 1, "factor": 5},
        96: {"seq_len": 384, "label_len": 384, "e_layers": 2, "d_layers": 1, "factor": 5},
        288: {"seq_len": 672, "label_len": 288, "e_layers": 2, "d_layers": 1, "factor": 5},
        672: {"seq_len": 672, "label_len": 384, "e_layers": 2, "d_layers": 1, "factor": 5},
    },
    "ETTm2": {
        24: {"seq_len": 672, "label_len": 96, "e_layers": 2, "d_layers": 1, "factor": 5},
        48: {"seq_len": 96, "label_len": 48, "e_layers": 2, "d_layers": 1, "factor": 5},
        96: {"seq_len": 384, "label_len": 384, "e_layers": 2, "d_layers": 1, "factor": 5},
        288: {"seq_len": 672, "label_len": 288, "e_layers": 2, "d_layers": 1, "factor": 5},
        672: {"seq_len": 672, "label_len": 384, "e_layers": 2, "d_layers": 1, "factor": 5},
    },
}

for solar_name in [f"Solar{i}" for i in range(1, 9)]:
    HORIZONS[solar_name] = {
        24: {"seq_len": 96, "label_len": 48, "e_layers": 2, "d_layers": 1, "factor": 5},
        96: {"seq_len": 96, "label_len": 48, "e_layers": 2, "d_layers": 1, "factor": 5},
    }

CONFIG_FLAGS = {
    "gelu_all": {
        "activation": "gelu",
        "encoder_activation": "gelu",
        "decoder_activation": "gelu",
        "output_activation": "linear",
        "amplitude": 0.0,
        "lee_type": 1,
    },
    "relu_all": {
        "activation": "relu",
        "encoder_activation": "relu",
        "decoder_activation": "relu",
        "output_activation": "linear",
        "amplitude": 0.0,
        "lee_type": 1,
    },
    "swish_all": {
        "activation": "swish",
        "encoder_activation": "swish",
        "decoder_activation": "swish",
        "output_activation": "linear",
        "amplitude": 0.0,
        "lee_type": 1,
    },
    "tanh_all": {
        "activation": "tanh",
        "encoder_activation": "tanh",
        "decoder_activation": "tanh",
        "output_activation": "linear",
        "amplitude": 0.0,
        "lee_type": 1,
    },
    "softsign_all": {
        "activation": "softsign",
        "encoder_activation": "softsign",
        "decoder_activation": "softsign",
        "output_activation": "linear",
        "amplitude": 0.0,
        "lee_type": 1,
    },
    "tanh_sin001_all": {
        "activation": "tanh_sin",
        "encoder_activation": "tanh_sin",
        "decoder_activation": "tanh_sin",
        "output_activation": "linear",
        "amplitude": 0.01,
        "frequency": 1.0,
        "phase": 0.0,
        "lee_type": 1,
    },
    "enc_tanh_dec_gelu": {
        "activation": "gelu",
        "encoder_activation": "tanh",
        "decoder_activation": "gelu",
        "output_activation": "linear",
        "amplitude": 0.0,
        "lee_type": 1,
    },
    "enc_gelu_dec_tanh": {
        "activation": "gelu",
        "encoder_activation": "gelu",
        "decoder_activation": "tanh",
        "output_activation": "linear",
        "amplitude": 0.0,
        "lee_type": 1,
    },
    "tanh_all_outtanh": {
        "activation": "tanh",
        "encoder_activation": "tanh",
        "decoder_activation": "tanh",
        "output_activation": "tanh",
        "amplitude": 0.0,
        "lee_type": 1,
    },
}

SUMMARY_GLOBS = [
    "phase2_remote_results/*/results/summary.csv",
    "phase3_remote_results/*/results/summary.csv",
    "phase4_remote_results/*/results/summary.csv",
    "phase4_remote_results/*/*/results/summary.csv",
]

PROTOCOL_COLUMNS = [
    "dataset",
    "pred_len",
    "seq_len",
    "label_len",
    "e_layers",
    "d_layers",
    "factor",
    "batch_size",
    "train_epochs",
    "activation_signature",
    "seed",
]

SETTING_RE = re.compile(
    r"_sl(?P<seq_len>\d+)_ll(?P<label_len>\d+)_pl(?P<pred_len_setting>\d+)"
    r"_dm(?P<d_model>\d+)_nh(?P<n_heads>\d+)_el(?P<e_layers>\d+)"
    r"_dl(?P<d_layers>\d+)_df(?P<d_ff>\d+)_fc(?P<factor>\d+)_seed(?P<seed_setting>\d+)_"
)


@dataclass(frozen=True)
class RunPaths:
    summary_path: str
    setting: str
    result_dir: str
    config_path: str
    pred_path: str
    true_path: str


def activation_signature(activation, encoder_activation, decoder_activation, output_activation, amplitude, lee_type):
    return (
        f"act={activation}|enc={encoder_activation}|dec={decoder_activation}|"
        f"out={output_activation}|a={float(amplitude):g}|lee={int(float(lee_type))}"
    )


def config_signature(config_name):
    config = CONFIG_FLAGS[config_name]
    return activation_signature(
        config["activation"],
        config["encoder_activation"],
        config["decoder_activation"],
        config["output_activation"],
        config.get("amplitude", 0.0),
        config.get("lee_type", 1),
    )


def config_from_signature(signature):
    reverse = {config_signature(name): name for name in CONFIG_NAMES}
    return reverse.get(signature, "")


def all_cells():
    rows = []
    for dataset in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"] + [f"Solar{i}" for i in range(1, 9)]:
        for pred_len, protocol in HORIZONS[dataset].items():
            rows.append({"dataset": dataset, "pred_len": pred_len, **protocol})
    return rows


def expected_matrix():
    rows = []
    for cell in all_cells():
        for config_name in CONFIG_NAMES:
            config = CONFIG_FLAGS[config_name]
            sig = config_signature(config_name)
            for seed in SEEDS:
                rows.append(
                    {
                        **cell,
                        **CANONICAL_PROTOCOL,
                        "config_name": config_name,
                        "activation": config["activation"],
                        "activation_family": activation_family(config["activation"]),
                        "encoder_activation": config["encoder_activation"],
                        "decoder_activation": config["decoder_activation"],
                        "output_activation": config["output_activation"],
                        "amplitude": config.get("amplitude", 0.0),
                        "frequency": config.get("frequency", 1.0),
                        "phase": config.get("phase", 0.0),
                        "lee_type": config.get("lee_type", 1),
                        "activation_signature": sig,
                        "seed": seed,
                        "target": target_for_dataset(cell["dataset"]),
                        "freq": freq_for_dataset(cell["dataset"]),
                        "data_path": data_path_for_dataset(cell["dataset"]),
                    }
                )
    return pd.DataFrame(rows)


def target_for_dataset(dataset):
    return "Power" if dataset.startswith("Solar") else "OT"


def freq_for_dataset(dataset):
    return "t" if dataset.startswith("Solar") or dataset.startswith("ETTm") else "h"


def root_path_for_dataset(dataset):
    return "../Solar" if dataset.startswith("Solar") else "./ETT-small"


def data_path_for_dataset(dataset):
    if dataset in ETT_FILES:
        return os.path.basename(ETT_FILES[dataset])
    return os.path.basename(SOLAR_FILES[dataset])


def result_paths(summary_path, setting):
    result_root = os.path.dirname(summary_path)
    result_dir = os.path.join(result_root, str(setting))
    return RunPaths(
        summary_path=summary_path,
        setting=str(setting),
        result_dir=result_dir,
        config_path=os.path.join(result_dir, "config.json"),
        pred_path=os.path.join(result_dir, "pred.npy"),
        true_path=os.path.join(result_dir, "true.npy"),
    )


def as_number(value, default=np.nan):
    try:
        if value == "" or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_scalar(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return ""
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return str(value)


def run_key(row):
    return tuple(normalize_scalar(row.get(column, "")) for column in PROTOCOL_COLUMNS)


def fill_summary_from_config(row):
    paths = result_paths(row["source_summary"], row.get("setting", ""))
    if os.path.exists(paths.config_path):
        try:
            with open(paths.config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        except (json.JSONDecodeError, OSError):
            config = {}
        for column in [
            "dataset",
            "pred_len",
            "seed",
            "activation",
            "encoder_activation",
            "decoder_activation",
            "output_activation",
            "activation_signature",
            "seq_len",
            "label_len",
            "d_model",
            "n_heads",
            "e_layers",
            "d_layers",
            "d_ff",
            "factor",
            "dropout",
            "attn",
            "embed",
            "batch_size",
            "train_epochs",
            "target",
            "features",
            "amplitude",
            "frequency",
            "phase",
            "lee_type",
            "data_path",
            "root_path",
        ]:
            if row.get(column, "") == "" or pd.isna(row.get(column, np.nan)):
                row[column] = config.get(column, row.get(column, ""))
        row["config_path"] = paths.config_path
        row["has_config"] = True
    else:
        row["config_path"] = ""
        row["has_config"] = False
    row["pred_path"] = paths.pred_path if os.path.exists(paths.pred_path) else ""
    row["true_path"] = paths.true_path if os.path.exists(paths.true_path) else ""
    row["has_arrays"] = bool(row["pred_path"] and row["true_path"] and row["has_config"])
    return row


def read_summary(path):
    frame = pd.read_csv(path)
    frame["source_summary"] = path
    frame["_source_order"] = np.arange(len(frame))
    for column in [
        "setting",
        "dataset",
        "activation",
        "encoder_activation",
        "decoder_activation",
        "output_activation",
        "activation_signature",
        "features",
        "target",
        "des",
        "run_id",
        "server_id",
        "server_ip",
        "attn",
        "embed",
    ]:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("")
    for column in [
        "pred_len",
        "seed",
        "seq_len",
        "label_len",
        "d_model",
        "n_heads",
        "e_layers",
        "d_layers",
        "d_ff",
        "factor",
        "dropout",
        "batch_size",
        "train_epochs",
        "amplitude",
        "frequency",
        "phase",
        "lee_type",
        "mse",
        "mae",
    ]:
        if column not in frame.columns:
            frame[column] = np.nan

    parsed = frame["setting"].fillna("").str.extract(SETTING_RE)
    for column in ["seq_len", "label_len", "d_model", "n_heads", "e_layers", "d_layers", "d_ff", "factor"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(
            pd.to_numeric(parsed[column], errors="coerce")
        )
    frame["pred_len"] = pd.to_numeric(frame["pred_len"], errors="coerce").fillna(
        pd.to_numeric(parsed["pred_len_setting"], errors="coerce")
    )
    frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").fillna(
        pd.to_numeric(parsed["seed_setting"], errors="coerce")
    )

    rows = []
    for row in frame.to_dict("records"):
        row = fill_summary_from_config(row)
        activation = row.get("activation", "")
        if not row.get("encoder_activation", ""):
            row["encoder_activation"] = activation
        if not row.get("decoder_activation", ""):
            row["decoder_activation"] = activation
        if not row.get("output_activation", ""):
            row["output_activation"] = "linear"
        if not row.get("activation_signature", ""):
            row["activation_signature"] = activation_signature(
                activation,
                row["encoder_activation"],
                row["decoder_activation"],
                row["output_activation"],
                as_number(row.get("amplitude", 0.0), 0.0),
                as_number(row.get("lee_type", 1), 1),
            )
        row["config_name"] = config_from_signature(row["activation_signature"])
        rows.append(row)
    return pd.DataFrame(rows)


def read_all_summaries(patterns=None):
    paths = []
    for pattern in patterns or SUMMARY_GLOBS:
        paths.extend(glob.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        return pd.DataFrame()
    frames = [read_summary(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def reusable_runs(expected, summaries):
    if summaries.empty:
        return pd.DataFrame(), expected.copy()

    exp = expected.copy()
    exp["_run_key"] = exp.apply(run_key, axis=1)
    candidate = summaries.copy()
    for column in [
        "pred_len",
        "seed",
        "seq_len",
        "label_len",
        "e_layers",
        "d_layers",
        "factor",
        "batch_size",
        "train_epochs",
    ]:
        candidate[column] = pd.to_numeric(candidate[column], errors="coerce")
    candidate = candidate[
        (candidate["config_name"] != "")
        & (candidate["has_config"].astype(bool))
        & candidate["mse"].notna()
        & candidate["mae"].notna()
        & (candidate["batch_size"] == CANONICAL_PROTOCOL["batch_size"])
        & (candidate["train_epochs"] == CANONICAL_PROTOCOL["train_epochs"])
        & (candidate["embed"] == CANONICAL_PROTOCOL["embed"])
        & (candidate["attn"] == CANONICAL_PROTOCOL["attn"])
        & (~candidate["des"].fillna("").str.contains("bs4|bs8|batch_adjusted|lee_bs", regex=True))
        & (~candidate["setting"].fillna("").str.contains("bs4|bs8|batch_adjusted|lee_bs", regex=True))
    ].copy()
    candidate["_run_key"] = candidate.apply(run_key, axis=1)
    candidate = candidate[candidate["_run_key"].isin(set(exp["_run_key"]))]
    candidate["_dedupe_order"] = np.arange(len(candidate))
    candidate["_has_arrays_order"] = candidate["has_arrays"].astype(bool).astype(int)
    candidate = (
        candidate.sort_values(["_has_arrays_order", "source_summary", "_source_order", "_dedupe_order"])
        .drop_duplicates("_run_key", keep="last")
        .copy()
    )
    reusable = exp.merge(
        candidate[
            [
                "_run_key",
                "setting",
                "source_summary",
                "config_path",
                "pred_path",
                "true_path",
                "has_arrays",
                "server_id",
                "server_ip",
                "run_id",
                "des",
                "mse",
                "mae",
            ]
        ],
        on="_run_key",
        how="inner",
    ).drop(columns=["_run_key"])
    missing = exp[~exp["_run_key"].isin(set(reusable.apply(run_key, axis=1)))].drop(columns=["_run_key"])
    return reusable, missing


def validate_local_data(project_root="."):
    rows = []
    required = {
        "ETTh1": {"target": "OT", "min_rows": 1},
        "ETTh2": {"target": "OT", "min_rows": 1},
        "ETTm1": {"target": "OT", "min_rows": 1},
        "ETTm2": {"target": "OT", "min_rows": 1},
        **{f"Solar{i}": {"target": "Power", "min_rows": 1} for i in range(1, 9)},
    }
    for dataset, rel_path in DATA_FILES.items():
        path = os.path.join(project_root, rel_path)
        row = {"dataset": dataset, "path": rel_path, "exists": os.path.exists(path)}
        if not row["exists"]:
            row.update({"ok": False, "rows": 0, "columns": "", "reason": "missing"})
            rows.append(row)
            continue
        try:
            frame = pd.read_csv(path)
            columns = frame.columns.tolist()
            target = required[dataset]["target"]
            total_nan = int(frame.isna().sum().sum())
            target_nan = int(frame[target].isna().sum()) if target in columns else -1
            schema_ok = "date" in columns and target in columns and len(frame) >= required[dataset]["min_rows"]
            ok = bool(schema_ok and total_nan == 0 and target_nan == 0)
            reasons = []
            if not schema_ok:
                reasons.append(f"need date/{target} and >= {required[dataset]['min_rows']} rows")
            if target_nan > 0:
                reasons.append(f"{target} has {target_nan} NaN")
            if total_nan > 0:
                reasons.append(f"file has {total_nan} total NaN")
            row.update(
                {
                    "ok": ok,
                    "rows": len(frame),
                    "columns": ",".join(columns),
                    "target_nan": target_nan,
                    "total_nan": total_nan,
                    "sha256": file_sha256(path),
                    "reason": "; ".join(reasons),
                }
            )
        except Exception as exc:
            row.update({"ok": False, "rows": 0, "columns": "", "reason": repr(exc)})
        rows.append(row)
    return pd.DataFrame(rows)


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
