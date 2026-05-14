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
