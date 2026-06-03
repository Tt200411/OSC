import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd


PROTOCOL_RE = re.compile(
    r"_sl(?P<seq_len>\d+)_ll(?P<label_len>\d+)_pl(?P<pred_len_setting>\d+)"
    r"_dm(?P<d_model>\d+)_nh(?P<n_heads>\d+)_el(?P<e_layers>\d+)"
    r"_dl(?P<d_layers>\d+)_df(?P<d_ff>\d+)_fc(?P<factor>\d+)_seed(?P<seed_setting>\d+)_"
)

INFORMER_TABLE2_MSE = {
    ("ETTh1", 24): 0.577,
    ("ETTh1", 48): 0.685,
    ("ETTh1", 168): 0.931,
    ("ETTh1", 336): 1.128,
    ("ETTh1", 720): 1.215,
    ("ETTh2", 24): 0.720,
    ("ETTh2", 48): 1.457,
    ("ETTh2", 168): 3.489,
    ("ETTh2", 336): 2.723,
    ("ETTh2", 720): 3.467,
}


PROTOCOL_COLUMNS = [
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
]


def expand_paths(paths):
    matches = []
    for path in paths:
        if os.path.isdir(path):
            matches.extend(glob.glob(os.path.join(path, "**", "summary.csv"), recursive=True))
        else:
            matches.extend(glob.glob(path))
    return sorted(set(matches))


def read_summaries(paths):
    frames = []
    for path in expand_paths(paths):
        frame = pd.read_csv(path)
        frame["source_summary"] = path
        frame["_source_order"] = np.arange(len(frame))
        frames.append(frame)
    if not frames:
        raise SystemExit("No summary.csv files found")
    return pd.concat(frames, ignore_index=True)


def default_signature(row):
    activation = value_or(row, "activation", "")
    enc = value_or(row, "encoder_activation", activation) or activation
    dec = value_or(row, "decoder_activation", activation) or activation
    out = value_or(row, "output_activation", "linear") or "linear"
    amplitude = as_float(row.get("amplitude", 0.0), 0.0)
    lee_type = int(as_float(row.get("lee_type", 1), 1))
    return f"act={activation}|enc={enc}|dec={dec}|out={out}|a={amplitude:g}|lee={lee_type}"


def value_or(row, key, default):
    value = row.get(key, default)
    if pd.isna(value) or value == "":
        return default
    return value


def as_float(value, default=np.nan):
    try:
        if value == "" or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=np.nan):
    number = as_float(value, default)
    if pd.isna(number):
        return default
    return int(number)


def normalize_columns(df):
    df = df.copy()
    for column in [
        "setting",
        "dataset",
        "features",
        "target",
        "activation",
        "encoder_activation",
        "decoder_activation",
        "output_activation",
        "activation_signature",
        "activation_family",
        "run_id",
        "des",
        "server_ip",
    ]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("")

    for column in PROTOCOL_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    df.loc[df["encoder_activation"] == "", "encoder_activation"] = df.loc[
        df["encoder_activation"] == "", "activation"
    ]
    df.loc[df["decoder_activation"] == "", "decoder_activation"] = df.loc[
        df["decoder_activation"] == "", "activation"
    ]
    df.loc[df["output_activation"] == "", "output_activation"] = "linear"
    missing_sig = df["activation_signature"] == ""
    df.loc[missing_sig, "activation_signature"] = df[missing_sig].apply(default_signature, axis=1)

    parsed = df["setting"].fillna("").str.extract(PROTOCOL_RE)
    for column in ["seq_len", "label_len", "d_model", "n_heads", "e_layers", "d_layers", "d_ff", "factor"]:
        parsed_column = pd.to_numeric(parsed[column], errors="coerce")
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(parsed_column)
    if "pred_len_setting" in parsed:
        df["pred_len"] = pd.to_numeric(df.get("pred_len", np.nan), errors="coerce").fillna(
            pd.to_numeric(parsed["pred_len_setting"], errors="coerce")
        )
    return fill_from_configs(df)


def fill_from_configs(df):
    rows = []
    for row in df.to_dict("records"):
        config_path = os.path.join(os.path.dirname(row["source_summary"]), str(row["setting"]), "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
            for column in PROTOCOL_COLUMNS:
                if pd.isna(row.get(column, np.nan)) or row.get(column, "") == "":
                    row[column] = config.get(column, row.get(column, np.nan))
            row["config_path"] = config_path
        else:
            row["config_path"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def activation_counterpart_signature(row, output_activation):
    amplitude = as_float(row.get("amplitude", 0.0), 0.0)
    lee_type = int(as_float(row.get("lee_type", 1), 1))
    return (
        f"act={row['activation']}|enc={row['encoder_activation']}|"
        f"dec={row['decoder_activation']}|out={output_activation}|a={amplitude:g}|lee={lee_type}"
    )


def protocol_key_columns():
    return [
        "dataset",
        "pred_len",
        "features",
        "target",
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
        "attn",
        "embed",
        "batch_size",
        "train_epochs",
    ]


def dedupe(df):
    df = df.copy()
    df["_global_order"] = np.arange(len(df))
    key_cols = protocol_key_columns() + ["activation_signature"]
    return (
        df.sort_values(["source_summary", "_source_order", "_global_order"])
        .drop_duplicates(key_cols, keep="last")
        .copy()
    )


def add_comparisons(df):
    df = df.copy()
    base_cols = protocol_key_columns()
    lookup = {
        tuple(row[col] for col in base_cols + ["activation_signature"]): row
        for row in df.to_dict("records")
    }
    gelu_sig = "act=gelu|enc=gelu|dec=gelu|out=linear|a=0|lee=1"
    gelu_mses = []
    gelu_improvements = []
    linear_mses = []
    linear_improvements = []
    for row in df.to_dict("records"):
        gelu_key = tuple(row[col] for col in base_cols) + (gelu_sig,)
        gelu_mse = as_float(lookup.get(gelu_key, {}).get("mse", np.nan))
        row_mse = as_float(row.get("mse", np.nan))
        gelu_mses.append(gelu_mse)
        gelu_improvements.append((gelu_mse - row_mse) / gelu_mse if gelu_mse and not pd.isna(gelu_mse) else np.nan)

        if row.get("output_activation") != "linear":
            linear_sig = activation_counterpart_signature(row, "linear")
            linear_key = tuple(row[col] for col in base_cols) + (linear_sig,)
            linear_mse = as_float(lookup.get(linear_key, {}).get("mse", np.nan))
        else:
            linear_mse = np.nan
        linear_mses.append(linear_mse)
        linear_improvements.append((linear_mse - row_mse) / linear_mse if linear_mse and not pd.isna(linear_mse) else np.nan)
    df["gelu_same_protocol_mse"] = gelu_mses
    df["relative_mse_improvement_vs_gelu"] = gelu_improvements
    df["linear_counterpart_mse"] = linear_mses
    df["relative_mse_improvement_vs_linear_counterpart"] = linear_improvements
    return df


def aggregate(df):
    group_cols = [
        "dataset",
        "pred_len",
        "seq_len",
        "label_len",
        "e_layers",
        "d_layers",
        "factor",
        "batch_size",
        "train_epochs",
        "dropout",
        "embed",
        "activation",
        "encoder_activation",
        "decoder_activation",
        "output_activation",
        "activation_signature",
    ]
    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            seed_count=("seed", "nunique"),
            run_count=("setting", "count"),
            improvement_vs_gelu_mean=("relative_mse_improvement_vs_gelu", "mean"),
            improvement_vs_gelu_min=("relative_mse_improvement_vs_gelu", "min"),
            win_rate_vs_gelu=("relative_mse_improvement_vs_gelu", lambda s: np.mean(np.asarray(s.dropna()) > 0) if s.dropna().size else np.nan),
        )
        .reset_index()
        .sort_values(["dataset", "pred_len", "mse_mean"])
    )


def informer_compare(agg):
    rows = []
    for row in agg.to_dict("records"):
        key = (row["dataset"], int(row["pred_len"]))
        if key not in INFORMER_TABLE2_MSE:
            continue
        informer_mse = INFORMER_TABLE2_MSE[key]
        rows.append(
            {
                **row,
                "informer_table2_mse": informer_mse,
                "relative_mse_improvement_vs_informer_table2": (informer_mse - row["mse_mean"]) / informer_mse,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["dataset", "pred_len", "relative_mse_improvement_vs_informer_table2"],
        ascending=[True, True, False],
    )


def contains_any(df, patterns, default=False):
    if not patterns:
        return pd.Series(default, index=df.index)
    searchable = (
        df["run_id"].fillna("").astype(str)
        + " "
        + df["des"].fillna("").astype(str)
        + " "
        + df["setting"].fillna("").astype(str)
    )
    mask = pd.Series(False, index=df.index)
    for pattern in patterns:
        mask |= searchable.str.contains(pattern, regex=False)
    return mask


def main():
    parser = argparse.ArgumentParser(description="Analyze protocol-aligned phase2 Informer activation results.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--include_run_contains", action="append", default=["phase2"])
    parser.add_argument("--exclude_run_contains", action="append", default=[])
    args = parser.parse_args()

    df = normalize_columns(read_summaries(args.paths))
    df = df[contains_any(df, args.include_run_contains, default=True)]
    df = df[~contains_any(df, args.exclude_run_contains, default=False)]

    deduped = add_comparisons(dedupe(df))
    agg = aggregate(deduped)
    table2 = informer_compare(agg)

    os.makedirs(args.output_dir, exist_ok=True)
    deduped.to_csv(os.path.join(args.output_dir, "phase2_protocol_raw_dedup.csv"), index=False)
    agg.to_csv(os.path.join(args.output_dir, "phase2_protocol_aggregate.csv"), index=False)
    table2.to_csv(os.path.join(args.output_dir, "phase2_informer_table2_compare.csv"), index=False)

    print(f"rows_raw={len(df)} rows_dedup={len(deduped)} aggregate_rows={len(agg)}")
    if not table2.empty:
        print(table2.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
