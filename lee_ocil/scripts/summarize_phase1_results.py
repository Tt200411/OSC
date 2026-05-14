import argparse
import glob
import os

import numpy as np
import pandas as pd


BASELINE_BY_ACTIVATION = {
    "gelu": "gelu",
    "relu": "relu",
    "tanh": "gelu",
    "softsign": "gelu",
    "scaled_tanh": "gelu",
    "gelu_sin": "gelu",
    "relu_sin": "relu",
    "tanh_sin": "tanh",
    "gelu_cos": "gelu",
    "relu_cos": "relu",
    "tanh_cos": "tanh",
    "tanh_rand": "tanh",
    "dynamic_gelu_sin": "gelu",
    "lee": "gelu",
}


def read_summaries(paths):
    frames = []
    for path in paths:
        if os.path.isdir(path):
            matches = glob.glob(os.path.join(path, "**", "summary.csv"), recursive=True)
        else:
            matches = glob.glob(path)
        for match in matches:
            frame = pd.read_csv(match)
            frame["source_summary"] = match
            frames.append(frame)
    if not frames:
        raise SystemExit("No summary.csv files found")
    return pd.concat(frames, ignore_index=True)


def normalize_columns(df):
    df = df.copy()
    for column in ["target", "server_ip", "des", "run_id"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("")
    return df


def split_filter_values(values):
    result = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                result.append(item)
    return result


def baseline_mapping(overrides):
    mapping = BASELINE_BY_ACTIVATION.copy()
    for override in split_filter_values(overrides):
        if "=" not in override:
            raise SystemExit(f"Invalid --baseline_override value: {override}")
        activation, baseline = [part.strip().lower() for part in override.split("=", 1)]
        if not activation or not baseline:
            raise SystemExit(f"Invalid --baseline_override value: {override}")
        mapping[activation] = baseline
    return mapping


def apply_filters(df, include_run_id, include_des, exclude_setting_contains, exclude_des_contains):
    df = df.copy()
    include_run_ids = split_filter_values(include_run_id)
    include_des_values = split_filter_values(include_des)
    if include_run_ids:
        df = df[df["run_id"].fillna("").astype(str).isin(include_run_ids)]
    if include_des_values:
        df = df[df["des"].fillna("").astype(str).isin(include_des_values)]
    for pattern in exclude_setting_contains:
        df = df[~df["setting"].fillna("").str.contains(pattern, regex=False)]
    for pattern in exclude_des_contains:
        df = df[~df["des"].fillna("").str.contains(pattern, regex=False)]
    return df


def add_baseline_columns(df, baseline_by_activation):
    df = df.copy()
    keys = ["dataset", "pred_len", "features", "target", "seed", "server_ip"]
    baseline_activations = sorted(set(baseline_by_activation.values()))
    baseline_rows = df[df["activation"].isin(baseline_activations)].copy()
    baseline_rows["baseline_activation"] = baseline_rows["activation"]
    baseline_rows = baseline_rows[keys + ["baseline_activation", "mse", "des", "run_id"]]
    baseline_rows = baseline_rows.rename(columns={"mse": "baseline_mse"})

    rows = []
    for _, row in df.iterrows():
        baseline_activation = baseline_by_activation.get(row["activation"], "gelu")
        row = row.to_dict()
        if row["activation"] == baseline_activation:
            baseline_mse = float(row["mse"])
        else:
            candidates = baseline_rows[baseline_rows["baseline_activation"] == baseline_activation]
            for key in keys:
                candidates = candidates[candidates[key] == row[key]]

            run_id = str(row.get("run_id", "") or "")
            if run_id:
                scoped = candidates[candidates["run_id"].fillna("").astype(str) == run_id]
                if not scoped.empty:
                    candidates = scoped

            des = str(row.get("des", "") or "")
            if des:
                scoped = candidates[candidates["des"].fillna("").astype(str) == des]
                if not scoped.empty:
                    candidates = scoped

            baseline_mse = np.nan if candidates.empty else float(candidates.iloc[0]["baseline_mse"])
        row["baseline_activation"] = baseline_activation
        row["baseline_mse"] = baseline_mse
        row["relative_mse_change"] = (
            (baseline_mse - row["mse"]) / baseline_mse if baseline_mse and not np.isnan(baseline_mse) else np.nan
        )
        row["win_against_baseline"] = (
            bool(row["mse"] < baseline_mse) if baseline_mse and not np.isnan(baseline_mse) else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(df):
    group_cols = [
        "dataset",
        "pred_len",
        "features",
        "activation",
        "activation_family",
        "amplitude",
        "frequency",
        "phase",
        "lee_type",
        "baseline_activation",
    ]
    grouped = df.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        seed_count=("seed", "nunique"),
        server_count=("server_ip", "nunique"),
        relative_mse_change_mean=("relative_mse_change", "mean"),
        win_rate=("win_against_baseline", "mean"),
    ).reset_index()
    summary["effect_label"] = np.select(
        [
            (summary["relative_mse_change_mean"] >= 0.05) & (summary["win_rate"] >= 0.8),
            (summary["relative_mse_change_mean"] >= 0.02) & (summary["win_rate"] >= 0.6),
            (summary["relative_mse_change_mean"] < -0.02),
            (summary["relative_mse_change_mean"] < 0.01) | (summary["win_rate"] < 0.5),
        ],
        ["strong_effective", "effective", "negative", "ineffective"],
        default="mixed",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize Phase 1 experiment results")
    parser.add_argument("paths", nargs="+", help="summary.csv files, glob patterns, or directories")
    parser.add_argument("--output_dir", default="analysis")
    parser.add_argument("--include_run_id", action="append", default=[])
    parser.add_argument("--include_des", action="append", default=[])
    parser.add_argument("--exclude_setting_contains", action="append", default=[])
    parser.add_argument("--exclude_des_contains", action="append", default=[])
    parser.add_argument(
        "--baseline_override",
        action="append",
        default=[],
        help="Override baseline mapping, e.g. --baseline_override lee=tanh",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    raw = normalize_columns(read_summaries(args.paths))
    raw = apply_filters(
        raw,
        args.include_run_id,
        args.include_des,
        args.exclude_setting_contains,
        args.exclude_des_contains,
    )
    detailed = add_baseline_columns(raw, baseline_mapping(args.baseline_override))
    summary = aggregate(detailed)

    detailed_path = os.path.join(args.output_dir, "phase1_detailed_results.csv")
    summary_path = os.path.join(args.output_dir, "phase1_activation_summary.csv")
    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {detailed_path}: {len(detailed)} rows")
    print(f"Wrote {summary_path}: {len(summary)} rows")


if __name__ == "__main__":
    main()
