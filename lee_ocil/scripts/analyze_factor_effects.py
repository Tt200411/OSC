import argparse
import json
import os

import numpy as np
import pandas as pd


BASELINE_BY_ACTIVATION = {
    "gelu": "gelu",
    "relu": "relu",
    "gelu_sin": "gelu",
    "relu_sin": "relu",
    "dynamic_gelu_sin": "gelu",
    "lee": "gelu",
}

FACTOR_COLUMNS = [
    "volatility",
    "anomaly_density_intensity",
    "turbulence_score",
    "mean",
    "std",
    "cv",
    "trend",
    "range",
    "mean_abs_change",
    "max_abs_change",
    "skewness",
    "kurtosis",
]


def split_filter_values(values):
    result = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                result.append(item)
    return result


def read_summary(path):
    frame = pd.read_csv(path)
    for column in ["target", "server_ip", "run_id", "des"]:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("")
    return frame


def apply_filters(df, include_run_id, include_des):
    run_ids = split_filter_values(include_run_id)
    des_values = split_filter_values(include_des)
    if run_ids:
        df = df[df["run_id"].fillna("").astype(str).isin(run_ids)]
    if des_values:
        df = df[df["des"].fillna("").astype(str).isin(des_values)]
    return df.copy()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_csv_path(config, lee_root):
    root_path = config.get("root_path", "")
    data_path = config.get("data_path", "")
    if os.path.isabs(data_path):
        return data_path
    if os.path.isabs(root_path):
        return os.path.join(root_path, data_path)
    return os.path.normpath(os.path.join(lee_root, root_path, data_path))


def test_border1(dataset, seq_len, config, lee_root):
    if dataset in {"ETTh1", "ETTh2"}:
        return 12 * 30 * 24 + 4 * 30 * 24 - seq_len
    if dataset in {"ETTm1", "ETTm2"}:
        return 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len

    csv_path = resolve_csv_path(config, lee_root)
    row_count = len(pd.read_csv(csv_path))
    num_test = int(row_count * 0.2)
    return row_count - num_test - seq_len


def factors_for_window(factors, start_index, end_index):
    matches = factors[
        (factors["end_index"] >= start_index) & (factors["start_index"] <= end_index)
    ]
    if matches.empty:
        return {column: np.nan for column in FACTOR_COLUMNS}
    return matches[FACTOR_COLUMNS].mean(numeric_only=True).to_dict()


def factor_rows_for_prediction(config, factors, n_samples, lee_root):
    seq_len = int(config["seq_len"])
    pred_len = int(config["pred_len"])
    dataset = config["data"]
    border1 = test_border1(dataset, seq_len, config, lee_root)

    rows = []
    for sample_index in range(n_samples):
        input_start = border1 + sample_index
        input_end = input_start + seq_len - 1
        target_start = input_end + 1
        target_end = target_start + pred_len - 1
        input_factors = factors_for_window(factors, input_start, input_end)
        target_factors = factors_for_window(factors, target_start, target_end)
        row = {
            "sample_index": sample_index,
            "input_start_index": input_start,
            "input_end_index": input_end,
            "target_start_index": target_start,
            "target_end_index": target_end,
        }
        row.update({f"input_{key}": value for key, value in input_factors.items()})
        row.update({f"target_{key}": value for key, value in target_factors.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def load_prediction_result(row, results_dir, factors_dir, lee_root):
    setting = row["setting"]
    folder = os.path.join(results_dir, setting)
    pred = np.load(os.path.join(folder, "pred.npy"))
    true = np.load(os.path.join(folder, "true.npy"))
    config = load_json(os.path.join(folder, "config.json"))
    factors_path = os.path.join(factors_dir, f"{row['dataset']}_factors.csv")
    factors = pd.read_csv(factors_path)

    all_mse = np.mean((pred - true) ** 2, axis=(1, 2))
    target_mse = np.mean((pred[:, :, -1] - true[:, :, -1]) ** 2, axis=1)
    factor_rows = factor_rows_for_prediction(config, factors, len(pred), lee_root)
    for column in row.index:
        factor_rows[column] = row[column]
    factor_rows["sample_mse_all"] = all_mse
    factor_rows["sample_mse_target"] = target_mse
    return factor_rows


def attach_prediction_rows(summary, results_dir, factors_dir, lee_root):
    frames = []
    for _, row in summary.iterrows():
        folder = os.path.join(results_dir, row["setting"])
        required = [
            os.path.join(folder, "pred.npy"),
            os.path.join(folder, "true.npy"),
            os.path.join(folder, "config.json"),
            os.path.join(factors_dir, f"{row['dataset']}_factors.csv"),
        ]
        missing = [path for path in required if not os.path.exists(path)]
        if missing:
            print(f"Skipping {row['setting']}: missing {missing[0]}")
            continue
        frames.append(load_prediction_result(row, results_dir, factors_dir, lee_root))
    if not frames:
        raise SystemExit("No prediction rows could be loaded")
    return pd.concat(frames, ignore_index=True)


def pair_against_baseline(samples):
    keys = ["dataset", "pred_len", "features", "target", "seed", "server_ip", "sample_index"]
    baseline_rows = samples[samples["activation"].isin(["gelu", "relu"])].copy()
    baseline_rows["baseline_activation"] = baseline_rows["activation"]
    baseline_rows = baseline_rows[
        keys
        + [
            "baseline_activation",
            "sample_mse_all",
            "sample_mse_target",
            "run_id",
            "des",
        ]
    ].rename(
        columns={
            "sample_mse_all": "baseline_mse_all",
            "sample_mse_target": "baseline_mse_target",
        }
    )

    rows = []
    for _, row in samples.iterrows():
        row = row.to_dict()
        baseline_activation = BASELINE_BY_ACTIVATION.get(row["activation"], "gelu")
        row["baseline_activation"] = baseline_activation
        if row["activation"] == baseline_activation:
            row["baseline_mse_all"] = row["sample_mse_all"]
            row["baseline_mse_target"] = row["sample_mse_target"]
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
            if candidates.empty:
                row["baseline_mse_all"] = np.nan
                row["baseline_mse_target"] = np.nan
            else:
                row["baseline_mse_all"] = float(candidates.iloc[0]["baseline_mse_all"])
                row["baseline_mse_target"] = float(candidates.iloc[0]["baseline_mse_target"])
        for metric in ["all", "target"]:
            baseline = row[f"baseline_mse_{metric}"]
            current = row[f"sample_mse_{metric}"]
            row[f"relative_{metric}_mse_change"] = (
                (baseline - current) / baseline if baseline and not np.isnan(baseline) else np.nan
            )
            row[f"win_{metric}_against_baseline"] = (
                bool(current < baseline) if baseline and not np.isnan(baseline) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def bin_series(series, factor_name):
    valid = series.dropna()
    if valid.nunique() < 3:
        return pd.Series(np.where(series <= valid.median(), "low", "high"), index=series.index)
    labels = ["down", "flat", "up"] if factor_name == "trend" else ["low", "mid", "high"]
    return pd.qcut(series.rank(method="first"), q=3, labels=labels)


def summarize_bins(samples):
    comparison = samples[
        samples["activation"] != samples["baseline_activation"]
    ].copy()
    rows = []
    for factor in FACTOR_COLUMNS:
        bin_column = f"input_{factor}_bin"
        comparison[bin_column] = bin_series(comparison[f"input_{factor}"], factor)
        group_cols = [
            "dataset",
            "pred_len",
            "features",
            "target",
            "activation",
            "activation_family",
            "amplitude",
            "lee_type",
            "baseline_activation",
            bin_column,
        ]
        grouped = comparison.groupby(group_cols, dropna=False, observed=False)
        for keys, group in grouped:
            row = dict(zip(group_cols, keys))
            row["factor"] = factor
            row["factor_bin"] = row.pop(bin_column)
            row["sample_count"] = len(group)
            row["seed_count"] = group["seed"].nunique()
            row["baseline_target_mse_mean"] = group["baseline_mse_target"].mean()
            row["activation_target_mse_mean"] = group["sample_mse_target"].mean()
            row["relative_target_mse_change_mean"] = group[
                "relative_target_mse_change"
            ].mean()
            row["target_win_rate"] = group["win_target_against_baseline"].mean()
            row["baseline_all_mse_mean"] = group["baseline_mse_all"].mean()
            row["activation_all_mse_mean"] = group["sample_mse_all"].mean()
            row["relative_all_mse_change_mean"] = group["relative_all_mse_change"].mean()
            row["all_win_rate"] = group["win_all_against_baseline"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Link Phase 1 prediction errors to factor bins")
    parser.add_argument("summary_csv")
    parser.add_argument("--results_dir", default="")
    parser.add_argument("--factors_dir", default="lee_ocil/factors")
    parser.add_argument("--lee_root", default="lee_ocil")
    parser.add_argument("--output_dir", default="analysis_factors")
    parser.add_argument("--include_run_id", action="append", default=[])
    parser.add_argument("--include_des", action="append", default=[])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results_dir = args.results_dir or os.path.dirname(args.summary_csv)
    summary = read_summary(args.summary_csv)
    summary = apply_filters(summary, args.include_run_id, args.include_des)
    samples = attach_prediction_rows(summary, results_dir, args.factors_dir, args.lee_root)
    detailed = pair_against_baseline(samples)
    bin_summary = summarize_bins(detailed)

    detailed_path = os.path.join(args.output_dir, "phase1_factor_sample_results.csv")
    summary_path = os.path.join(args.output_dir, "phase1_factor_bin_summary.csv")
    detailed.to_csv(detailed_path, index=False)
    bin_summary.to_csv(summary_path, index=False)
    print(f"Wrote {detailed_path}: {len(detailed)} rows")
    print(f"Wrote {summary_path}: {len(bin_summary)} rows")


if __name__ == "__main__":
    main()
