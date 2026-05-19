import argparse
import os

import numpy as np
import pandas as pd


DATASETS = {
    "ETTh1": {
        "path": "ETT-small/ETTh1.csv",
        "target": "OT",
        "points_per_day": 24,
    },
    "ETTh2": {
        "path": "ETT-small/ETTh2.csv",
        "target": "OT",
        "points_per_day": 24,
    },
    "Solar": {
        "path": "../Solar/PV_Solar_Station_1.csv",
        "target": "Power",
        "points_per_day": 96,
    },
    "Solar1": {
        "path": "../Solar/Site_1_50MW.csv",
        "target": "Power",
        "points_per_day": 96,
    },
    "Solar5": {
        "path": "../Solar/Site_5_110MW.csv",
        "target": "Power",
        "points_per_day": 96,
    },
}


def safe_skew(values):
    std = values.std(ddof=0)
    if std < 1e-12:
        return 0.0
    z = (values - values.mean()) / std
    return float(np.mean(z ** 3))


def safe_kurtosis(values):
    std = values.std(ddof=0)
    if std < 1e-12:
        return 0.0
    z = (values - values.mean()) / std
    return float(np.mean(z ** 4) - 3.0)


def safe_lag_autocorr(values, lag=1):
    if len(values) <= lag:
        return 0.0
    left = values[:-lag]
    right = values[lag:]
    if left.std(ddof=0) < 1e-12 or right.std(ddof=0) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def sign_entropy(diffs):
    if len(diffs) == 0:
        return 0.0
    signs = np.sign(diffs)
    counts = np.array([(signs < 0).mean(), (signs == 0).mean(), (signs > 0).mean()])
    counts = counts[counts > 0]
    entropy = -np.sum(counts * np.log(counts))
    return float(entropy / np.log(3.0))


def max_drawdown_drawup(values):
    if len(values) == 0:
        return 0.0, 0.0
    running_max = np.maximum.accumulate(values)
    running_min = np.minimum.accumulate(values)
    drawdown = running_max - values
    drawup = values - running_min
    return float(drawdown.max()), float(drawup.max())


def window_factors(values, reference_values, alpha, beta, method):
    values = np.asarray(values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    diffs = np.diff(values)
    abs_diffs = np.abs(diffs)
    std = float(values.std(ddof=0))
    mean = float(values.mean())
    ref_std = float(reference_values.std(ddof=0))
    ref_mean = float(reference_values.mean())
    ref_abs_diffs = np.abs(np.diff(reference_values))
    ref_diff_std = float(np.diff(reference_values).std(ddof=0)) if len(reference_values) > 1 else 0.0
    volatility = std / (abs(ref_mean) + 1e-8)
    z = (values - ref_mean) / (ref_std + 1e-8)
    anomaly_mask = np.abs(z) > 2.0
    anomaly_density = float(anomaly_mask.mean())
    anomaly_intensity = float(np.mean(np.maximum(np.abs(z[anomaly_mask]) - 2.0, 0.0))) if anomaly_mask.any() else 0.0
    anomaly_density_intensity = anomaly_density * anomaly_intensity
    trend = float(np.polyfit(np.arange(len(values)), values, 1)[0]) if len(values) > 1 else 0.0
    mean_abs_change = float(abs_diffs.mean()) if len(abs_diffs) else 0.0
    max_abs_change = float(abs_diffs.max()) if len(abs_diffs) else 0.0
    ref_mean_abs_change = float(ref_abs_diffs.mean()) if len(ref_abs_diffs) else 0.0
    realized_volatility = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) else 0.0
    half = max(1, len(diffs) // 2)
    early_vol = float(np.sqrt(np.mean(diffs[:half] ** 2))) if len(diffs[:half]) else 0.0
    late_vol = float(np.sqrt(np.mean(diffs[half:] ** 2))) if len(diffs[half:]) else early_vol
    volatility_shock = (realized_volatility - ref_diff_std) / (ref_diff_std + 1e-8)
    positive_diffs = diffs[diffs > 0]
    negative_diffs = diffs[diffs < 0]
    upside_volatility = float(np.sqrt(np.mean(positive_diffs ** 2))) if len(positive_diffs) else 0.0
    downside_volatility = float(np.sqrt(np.mean(negative_diffs ** 2))) if len(negative_diffs) else 0.0
    cumulative_change = float(values[-1] - values[0]) if len(values) > 1 else 0.0
    recent = diffs[-max(1, len(diffs) // 4):] if len(diffs) else np.array([])
    momentum = float(recent.sum()) if len(recent) else 0.0
    sign = np.sign(diffs)
    nonzero_sign = sign[sign != 0]
    if len(nonzero_sign):
        trend_consistency = float(max((nonzero_sign > 0).mean(), (nonzero_sign < 0).mean()))
    else:
        trend_consistency = 0.0
    reversal_count = int(np.sum(nonzero_sign[1:] * nonzero_sign[:-1] < 0)) if len(nonzero_sign) > 1 else 0
    reversal_rate = float(reversal_count / max(len(nonzero_sign) - 1, 1)) if len(nonzero_sign) else 0.0
    centered = values - mean
    centered_sign = np.sign(centered)
    nonzero_centered = centered_sign[centered_sign != 0]
    mean_crossing_count = int(np.sum(nonzero_centered[1:] * nonzero_centered[:-1] < 0)) if len(nonzero_centered) > 1 else 0
    mean_crossing_rate = float(mean_crossing_count / max(len(nonzero_centered) - 1, 1)) if len(nonzero_centered) else 0.0
    lag1_autocorr = safe_lag_autocorr(diffs, lag=1)
    mean_reversion_strength = float(mean_crossing_rate + max(0.0, -lag1_autocorr))
    jump_threshold = max(2.0 * ref_diff_std, ref_mean_abs_change + 2.0 * ref_diff_std, 1e-8)
    jump_intensity = float((abs_diffs > jump_threshold).mean()) if len(abs_diffs) else 0.0
    median_abs_change = float(np.median(abs_diffs)) if len(abs_diffs) else 0.0
    tail_ratio = float(np.percentile(abs_diffs, 95) / (median_abs_change + 1e-8)) if len(abs_diffs) else 0.0
    max_drawdown, max_drawup = max_drawdown_drawup(values)
    volatility_component = std / (ref_std + 1e-8)
    anomaly_component = anomaly_density + anomaly_intensity
    change_component = mean_abs_change / (ref_mean_abs_change + 1e-8)
    components = [volatility_component, alpha * anomaly_component, beta * change_component]
    turbulence_score = float(sum(components) if method == "sum" else np.mean(components))
    return {
        "volatility": volatility,
        "anomaly_density_intensity": anomaly_density_intensity,
        "turbulence_score": turbulence_score,
        "mean": mean,
        "std": std,
        "cv": float(std / (abs(mean) + 1e-8)),
        "trend": trend,
        "range": float(values.max() - values.min()) if len(values) else 0.0,
        "mean_abs_change": mean_abs_change,
        "max_abs_change": max_abs_change,
        "skewness": safe_skew(values),
        "kurtosis": safe_kurtosis(values),
        "realized_volatility": realized_volatility,
        "volatility_shock": volatility_shock,
        "volatility_ratio_late_early": float(late_vol / (early_vol + 1e-8)),
        "upside_volatility": upside_volatility,
        "downside_volatility": downside_volatility,
        "cumulative_change": cumulative_change,
        "momentum": momentum,
        "trend_consistency": trend_consistency,
        "reversal_rate": reversal_rate,
        "mean_crossing_rate": mean_crossing_rate,
        "lag1_autocorr": lag1_autocorr,
        "mean_reversion_strength": mean_reversion_strength,
        "jump_intensity": jump_intensity,
        "tail_ratio": tail_ratio,
        "max_drawdown": max_drawdown,
        "max_drawup": max_drawup,
        "direction_entropy": sign_entropy(diffs),
    }


def build_factor_table(dataset, csv_path, target, points_per_day, reference_days, alpha, beta, method):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    values = df[target].astype(float).to_numpy()
    reference_size = reference_days * points_per_day
    rows = []
    for start in range(0, len(df), points_per_day):
        end = min(start + points_per_day, len(df))
        if end - start < points_per_day:
            continue
        ref_start = max(0, start - reference_size)
        reference = values[ref_start:start] if start > 0 else values[start:end]
        factors = window_factors(values[start:end], reference, alpha, beta, method)
        rows.append({
            "dataset": dataset,
            "window_id": len(rows),
            "start_index": start,
            "end_index": end - 1,
            "start_time": df["date"].iloc[start].isoformat(),
            "end_time": df["date"].iloc[end - 1].isoformat(),
            "points_per_day": points_per_day,
            "reference_days": reference_days,
            "alpha": alpha,
            "beta": beta,
            "method": method,
            **factors,
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate Phase 1 factor CSV files")
    parser.add_argument("--dataset", choices=DATASETS.keys(), action="append")
    parser.add_argument("--output_dir", default="factors")
    parser.add_argument("--reference_days", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--method", choices=["sum", "mean"], default="sum")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    selected = args.dataset or list(DATASETS.keys())
    all_frames = []
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    for dataset in selected:
        meta = DATASETS[dataset]
        csv_path = os.path.normpath(os.path.join(root, meta["path"]))
        factors = build_factor_table(
            dataset=dataset,
            csv_path=csv_path,
            target=meta["target"],
            points_per_day=meta["points_per_day"],
            reference_days=args.reference_days,
            alpha=args.alpha,
            beta=args.beta,
            method=args.method,
        )
        output_path = os.path.join(args.output_dir, f"{dataset}_factors.csv")
        factors.to_csv(output_path, index=False)
        all_frames.append(factors)
        print(f"Wrote {output_path}: {len(factors)} rows")

    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = os.path.join(args.output_dir, "all_factors.csv")
    combined.to_csv(combined_path, index=False)
    print(f"Wrote {combined_path}: {len(combined)} rows")


if __name__ == "__main__":
    main()
