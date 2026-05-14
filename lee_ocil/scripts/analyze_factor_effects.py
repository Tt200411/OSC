import argparse
import json
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

BIN_ORDER = {
    "low": 0,
    "mid": 1,
    "high": 2,
    "down": 0,
    "flat": 1,
    "up": 2,
}


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


def prepare_factor_lookup(factors):
    factors = factors.sort_values("start_index").reset_index(drop=True)
    starts = factors["start_index"].to_numpy()
    ends = factors["end_index"].to_numpy()
    values = factors[FACTOR_COLUMNS].to_numpy(dtype=float)
    valid = ~np.isnan(values)
    value_sums = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(np.where(valid, values, 0.0), axis=0)])
    value_counts = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(valid.astype(float), axis=0)])
    return starts, ends, value_sums, value_counts


def factor_matrix_for_windows(factor_lookup, start_indices, end_indices):
    starts, ends, value_sums, value_counts = factor_lookup
    left = np.searchsorted(ends, start_indices, side="left")
    right = np.searchsorted(starts, end_indices, side="right")
    output = np.full((len(start_indices), len(FACTOR_COLUMNS)), np.nan, dtype=float)
    has_overlap = left < right
    if has_overlap.any():
        row_sums = value_sums[right[has_overlap]] - value_sums[left[has_overlap]]
        row_counts = value_counts[right[has_overlap]] - value_counts[left[has_overlap]]
        means = np.full_like(row_sums, np.nan, dtype=float)
        np.divide(row_sums, row_counts, out=means, where=row_counts > 0)
        output[has_overlap] = means
    return output


def factor_rows_for_prediction(config, factors, n_samples, lee_root):
    seq_len = int(config["seq_len"])
    pred_len = int(config["pred_len"])
    dataset = config["data"]
    border1 = test_border1(dataset, seq_len, config, lee_root)

    sample_index = np.arange(n_samples)
    input_start = border1 + sample_index
    input_end = input_start + seq_len - 1
    target_start = input_end + 1
    target_end = target_start + pred_len - 1

    factor_lookup = prepare_factor_lookup(factors)
    input_factors = factor_matrix_for_windows(factor_lookup, input_start, input_end)
    target_factors = factor_matrix_for_windows(factor_lookup, target_start, target_end)

    rows = pd.DataFrame(
        {
            "sample_index": sample_index,
            "input_start_index": input_start,
            "input_end_index": input_end,
            "target_start_index": target_start,
            "target_end_index": target_end,
        }
    )
    for column_index, column in enumerate(FACTOR_COLUMNS):
        rows[f"input_{column}"] = input_factors[:, column_index]
        rows[f"target_{column}"] = target_factors[:, column_index]
    return rows


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


def pair_against_baseline(samples, baseline_by_activation):
    keys = ["dataset", "pred_len", "features", "target", "seed", "server_ip", "sample_index"]
    samples = samples.copy().reset_index(drop=True)
    samples["_row_id"] = np.arange(len(samples))
    samples["baseline_activation"] = (
        samples["activation"].map(baseline_by_activation).fillna("gelu")
    )

    baseline_activations = sorted(set(baseline_by_activation.values()))
    baseline_rows = samples[samples["activation"].isin(baseline_activations)].copy()
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
            "run_id": "baseline_run_id",
            "des": "baseline_des",
        }
    )

    merge_columns = ["_row_id"] + keys + [
        "activation",
        "baseline_activation",
        "sample_mse_all",
        "sample_mse_target",
        "run_id",
        "des",
    ]
    candidates = samples[merge_columns].merge(
        baseline_rows,
        on=keys + ["baseline_activation"],
        how="left",
    )

    row_run_id = candidates["run_id"].fillna("").astype(str)
    baseline_run_id = candidates["baseline_run_id"].fillna("").astype(str)
    row_des = candidates["des"].fillna("").astype(str)
    baseline_des = candidates["baseline_des"].fillna("").astype(str)

    same_run = (row_run_id != "") & (row_run_id == baseline_run_id)
    has_same_run = same_run.groupby(candidates["_row_id"]).transform("any")
    valid_after_run = ~has_same_run | same_run

    same_des = (row_des != "") & (row_des == baseline_des)
    has_same_des = (valid_after_run & same_des).groupby(candidates["_row_id"]).transform("any")
    valid_candidate = valid_after_run & (~has_same_des | same_des)
    valid_candidate &= candidates["baseline_mse_all"].notna()

    candidates["_valid_candidate"] = valid_candidate
    candidates["_candidate_order"] = candidates.groupby("_row_id").cumcount()
    selected = candidates.sort_values(
        ["_row_id", "_valid_candidate", "_candidate_order"],
        ascending=[True, False, True],
    ).drop_duplicates("_row_id")

    result = samples.drop(columns=["_row_id"]).copy()
    result = result.merge(
        selected[["_row_id", "baseline_mse_all", "baseline_mse_target"]],
        left_index=True,
        right_on="_row_id",
        how="left",
    ).sort_values("_row_id")
    result = result.drop(columns=["_row_id"]).reset_index(drop=True)

    self_baseline = result["activation"] == result["baseline_activation"]
    result.loc[self_baseline, "baseline_mse_all"] = result.loc[self_baseline, "sample_mse_all"]
    result.loc[self_baseline, "baseline_mse_target"] = result.loc[
        self_baseline, "sample_mse_target"
    ]

    for metric in ["all", "target"]:
        baseline = result[f"baseline_mse_{metric}"]
        current = result[f"sample_mse_{metric}"]
        valid = baseline.notna() & (baseline != 0)
        result[f"relative_{metric}_mse_change"] = np.nan
        result.loc[valid, f"relative_{metric}_mse_change"] = (
            baseline.loc[valid] - current.loc[valid]
        ) / baseline.loc[valid]
        win_column = f"win_{metric}_against_baseline"
        result[win_column] = pd.Series(pd.NA, index=result.index, dtype="boolean")
        result.loc[valid, win_column] = (current.loc[valid] < baseline.loc[valid]).to_numpy()
    return result


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
            row["target_mse_delta_mean"] = (
                row["baseline_target_mse_mean"] - row["activation_target_mse_mean"]
            )
            row["aggregate_relative_target_mse_change"] = (
                row["target_mse_delta_mean"] / row["baseline_target_mse_mean"]
                if row["baseline_target_mse_mean"]
                and not pd.isna(row["baseline_target_mse_mean"])
                else np.nan
            )
            row["relative_target_mse_change_mean"] = group[
                "relative_target_mse_change"
            ].mean()
            row["target_win_rate"] = group["win_target_against_baseline"].mean()
            row["baseline_all_mse_mean"] = group["baseline_mse_all"].mean()
            row["activation_all_mse_mean"] = group["sample_mse_all"].mean()
            row["all_mse_delta_mean"] = (
                row["baseline_all_mse_mean"] - row["activation_all_mse_mean"]
            )
            row["aggregate_relative_all_mse_change"] = (
                row["all_mse_delta_mean"] / row["baseline_all_mse_mean"]
                if row["baseline_all_mse_mean"]
                and not pd.isna(row["baseline_all_mse_mean"])
                else np.nan
            )
            row["relative_all_mse_change_mean"] = group["relative_all_mse_change"].mean()
            row["all_win_rate"] = group["win_all_against_baseline"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def safe_corr(left, right):
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 3:
        return np.nan
    if frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return np.nan
    return float(np.corrcoef(frame["left"], frame["right"])[0, 1])


def safe_spearman(left, right):
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 3:
        return np.nan
    return safe_corr(frame["left"].rank(method="average"), frame["right"].rank(method="average"))


def slope_from_bins(bin_values, metric):
    points = [
        (BIN_ORDER[str(row["factor_bin"])], row[metric])
        for _, row in bin_values.iterrows()
        if str(row["factor_bin"]) in BIN_ORDER and not pd.isna(row[metric])
    ]
    if len(points) < 2:
        return np.nan
    x = np.array([point[0] for point in points], dtype=float)
    y = np.array([point[1] for point in points], dtype=float)
    if len(np.unique(x)) < 2:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def value_for_bin(bin_values, metric, bin_name):
    match = bin_values[bin_values["factor_bin"].astype(str) == bin_name]
    if match.empty:
        return np.nan
    return float(match.iloc[0][metric])


def edge_bins_for_factor(factor):
    if factor == "trend":
        return "down", "flat", "up"
    return "low", "mid", "high"


def summarize_factor_contrasts(samples, bin_summary):
    comparison = samples[samples["activation"] != samples["baseline_activation"]].copy()
    for factor in FACTOR_COLUMNS:
        comparison[f"input_{factor}_bin"] = bin_series(comparison[f"input_{factor}"], factor)

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
        "factor",
    ]
    sample_group_cols = group_cols[:-1]
    rows = []
    for keys, bin_values in bin_summary.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        factor = row["factor"]
        start_bin, middle_bin, end_bin = edge_bins_for_factor(factor)

        sample_candidates = comparison
        for key, value in zip(sample_group_cols, keys[:-1]):
            sample_candidates = sample_candidates[sample_candidates[key] == value]

        row["sample_count"] = int(bin_values["sample_count"].sum())
        row["seed_count"] = int(bin_values["seed_count"].max())
        row["start_bin"] = start_bin
        row["middle_bin"] = middle_bin
        row["end_bin"] = end_bin
        for metric in [
            "target_win_rate",
            "aggregate_relative_target_mse_change",
            "relative_target_mse_change_mean",
            "target_mse_delta_mean",
            "all_win_rate",
            "aggregate_relative_all_mse_change",
            "relative_all_mse_change_mean",
            "all_mse_delta_mean",
        ]:
            start_value = value_for_bin(bin_values, metric, start_bin)
            middle_value = value_for_bin(bin_values, metric, middle_bin)
            end_value = value_for_bin(bin_values, metric, end_bin)
            row[f"{metric}_start"] = start_value
            row[f"{metric}_middle"] = middle_value
            row[f"{metric}_end"] = end_value
            row[f"{metric}_end_minus_start"] = (
                end_value - start_value
                if not pd.isna(end_value) and not pd.isna(start_value)
                else np.nan
            )
            row[f"{metric}_end_minus_middle"] = (
                end_value - middle_value
                if not pd.isna(end_value) and not pd.isna(middle_value)
                else np.nan
            )
            row[f"{metric}_spread"] = float(bin_values[metric].max() - bin_values[metric].min())
            row[f"{metric}_slope"] = slope_from_bins(bin_values, metric)

        factor_values = sample_candidates[f"input_{factor}"]
        row["relative_target_mse_change_spearman"] = safe_spearman(
            factor_values, sample_candidates["relative_target_mse_change"]
        )
        row["target_win_spearman"] = safe_spearman(
            factor_values, sample_candidates["win_target_against_baseline"].astype(float)
        )
        row["relative_all_mse_change_spearman"] = safe_spearman(
            factor_values, sample_candidates["relative_all_mse_change"]
        )
        row["all_win_spearman"] = safe_spearman(
            factor_values, sample_candidates["win_all_against_baseline"].astype(float)
        )

        target_metric = "relative_target_mse_change_mean"
        best = bin_values.loc[bin_values[target_metric].idxmax()]
        worst = bin_values.loc[bin_values[target_metric].idxmin()]
        row["best_target_bin"] = best["factor_bin"]
        row["worst_target_bin"] = worst["factor_bin"]
        aggregate_target_metric = "aggregate_relative_target_mse_change"
        aggregate_best = bin_values.loc[bin_values[aggregate_target_metric].idxmax()]
        aggregate_worst = bin_values.loc[bin_values[aggregate_target_metric].idxmin()]
        row["aggregate_best_target_bin"] = aggregate_best["factor_bin"]
        row["aggregate_worst_target_bin"] = aggregate_worst["factor_bin"]
        row["factor_conditioned"] = (
            abs(row["relative_target_mse_change_mean_slope"]) >= 0.03
            or abs(row["relative_target_mse_change_spearman"]) >= 0.10
            or row["relative_target_mse_change_mean_spread"] >= 0.08
        )
        row["aggregate_factor_conditioned"] = (
            abs(row["aggregate_relative_target_mse_change_slope"]) >= 0.03
            or row["aggregate_relative_target_mse_change_spread"] >= 0.08
        )
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
    parser.add_argument(
        "--baseline_override",
        action="append",
        default=[],
        help="Override baseline mapping, e.g. --baseline_override lee=tanh",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results_dir = args.results_dir or os.path.dirname(args.summary_csv)
    summary = read_summary(args.summary_csv)
    summary = apply_filters(summary, args.include_run_id, args.include_des)
    samples = attach_prediction_rows(summary, results_dir, args.factors_dir, args.lee_root)
    detailed = pair_against_baseline(samples, baseline_mapping(args.baseline_override))
    bin_summary = summarize_bins(detailed)
    contrast_summary = summarize_factor_contrasts(detailed, bin_summary)

    detailed_path = os.path.join(args.output_dir, "phase1_factor_sample_results.csv")
    summary_path = os.path.join(args.output_dir, "phase1_factor_bin_summary.csv")
    contrast_path = os.path.join(args.output_dir, "phase1_factor_contrast_summary.csv")
    detailed.to_csv(detailed_path, index=False)
    bin_summary.to_csv(summary_path, index=False)
    contrast_summary.to_csv(contrast_path, index=False)
    print(f"Wrote {detailed_path}: {len(detailed)} rows")
    print(f"Wrote {summary_path}: {len(bin_summary)} rows")
    print(f"Wrote {contrast_path}: {len(contrast_summary)} rows")


if __name__ == "__main__":
    main()
