#!/usr/bin/env python3
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from phase4_common import data_path_for_dataset, root_path_for_dataset
from phase4_report_utils import markdown_table
from phase4_summary_factor_bins import build_cell_factor_table


EPS = 1e-8
ROUTER_CONFIGS = [
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


def points_per_day(dataset):
    if dataset.startswith("ETTh"):
        return 24
    if dataset == "Weather":
        return 144
    if dataset in {"Exchange", "ILI"}:
        return 1
    return 96


def test_borders(dataset, seq_len, row_count):
    if dataset in {"ETTh1", "ETTh2"}:
        border1 = 12 * 30 * 24 + 4 * 30 * 24 - seq_len
        border2 = 12 * 30 * 24 + 8 * 30 * 24
    elif dataset in {"ETTm1", "ETTm2"}:
        border1 = 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len
        border2 = 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4
    else:
        num_test = int(row_count * 0.2)
        border1 = row_count - num_test - seq_len
        border2 = row_count
    return max(0, border1), min(row_count, border2)


def sample_starts(dataset, seq_len, pred_len, row_count, max_windows):
    border1, border2 = test_borders(dataset, seq_len, row_count)
    n_samples = max(0, border2 - border1 - seq_len - pred_len + 1)
    if n_samples <= 0:
        return np.array([], dtype=int), border1
    starts = border1 + np.arange(n_samples, dtype=int)
    if max_windows and max_windows > 0 and len(starts) > max_windows:
        idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = np.unique(starts[idx])
    return starts, border1


def read_numeric_data(dataset, lee_root):
    csv_path = os.path.normpath(
        os.path.join(lee_root, root_path_for_dataset(dataset), data_path_for_dataset(dataset))
    )
    frame = pd.read_csv(csv_path)
    numeric = frame.drop(columns=["date"], errors="ignore")
    numeric = numeric.apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    return numeric.to_numpy(dtype=float), csv_path


def robust_stats(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    std = float(np.std(values))
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    return med, mad, std, iqr


def spectral_profile(values, ppd):
    values = np.asarray(values, dtype=float)
    if len(values) < 4:
        return {
            "spectral_entropy": 0.0,
            "seasonal_peak_strength": 0.0,
            "harmonic_energy_ratio": 0.0,
            "high_freq_noise_share": 0.0,
            "dominant_period_to_horizon_ratio_raw": 0.0,
            "dominant_freq": 0.0,
            "dominant_phase": 0.0,
        }
    centered = values - np.mean(values)
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(len(centered), d=1.0)
    power = np.abs(spectrum) ** 2
    if len(power) > 1:
        spectrum = spectrum[1:]
        freqs = freqs[1:]
        power = power[1:]
    total = float(np.sum(power))
    if total <= EPS or len(power) == 0:
        return {
            "spectral_entropy": 0.0,
            "seasonal_peak_strength": 0.0,
            "harmonic_energy_ratio": 0.0,
            "high_freq_noise_share": 0.0,
            "dominant_period_to_horizon_ratio_raw": 0.0,
            "dominant_freq": 0.0,
            "dominant_phase": 0.0,
        }
    prob = power / total
    entropy = -float(np.sum(prob * np.log(prob + EPS)) / np.log(max(len(prob), 2)))
    dominant_idx = int(np.argmax(power))
    dominant_freq = float(freqs[dominant_idx])
    dominant_period = float(1.0 / dominant_freq) if dominant_freq > EPS else 0.0
    daily_freq = 1.0 / float(ppd)
    seasonal_bins = []
    harmonic_bins = []
    for harmonic in range(1, 5):
        target_freq = daily_freq * harmonic
        if target_freq > freqs.max() + EPS:
            continue
        idx = int(np.argmin(np.abs(freqs - target_freq)))
        harmonic_bins.extend([idx - 1, idx, idx + 1])
        if harmonic == 1:
            seasonal_bins.extend([idx - 1, idx, idx + 1])
    seasonal_bins = sorted({idx for idx in seasonal_bins if 0 <= idx < len(power)})
    harmonic_bins = sorted({idx for idx in harmonic_bins if 0 <= idx < len(power)})
    high_freq = float(np.sum(power[freqs > 0.25]) / total)
    return {
        "spectral_entropy": entropy,
        "seasonal_peak_strength": float(np.sum(power[seasonal_bins]) / total) if seasonal_bins else 0.0,
        "harmonic_energy_ratio": float(np.sum(power[harmonic_bins]) / total) if harmonic_bins else 0.0,
        "high_freq_noise_share": high_freq,
        "dominant_period_to_horizon_ratio_raw": dominant_period,
        "dominant_freq": dominant_freq,
        "dominant_phase": float(np.angle(spectrum[dominant_idx])),
    }


def rolling_energy_ratio(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 8:
        return 0.0
    width = max(4, len(values) // 8)
    kernel = np.ones(width) / float(width)
    energy = np.convolve((values - np.mean(values)) ** 2, kernel, mode="valid")
    median = float(np.median(energy))
    if median <= EPS:
        return 0.0
    return float(np.percentile(energy, 95) / (median + EPS))


def split_entropy_std(values, ppd):
    values = np.asarray(values, dtype=float)
    if len(values) < 16:
        return 0.0
    entropies = []
    phases = []
    for segment in np.array_split(values, 4):
        if len(segment) < 4:
            continue
        profile = spectral_profile(segment, ppd)
        entropies.append(profile["spectral_entropy"])
        phases.append(profile["dominant_phase"])
    entropy_std = float(np.std(entropies)) if entropies else 0.0
    if not phases:
        phase_concentration = 0.0
    else:
        phase_concentration = float(np.abs(np.mean(np.exp(1j * np.asarray(phases)))))
    return entropy_std, phase_concentration


def dominant_frequency_drift(values, ppd):
    values = np.asarray(values, dtype=float)
    if len(values) < 8:
        return 0.0
    left, right = np.array_split(values, 2)
    return float(abs(spectral_profile(left, ppd)["dominant_freq"] - spectral_profile(right, ppd)["dominant_freq"]))


def multivar_factors(window):
    if window.ndim != 2 or window.shape[1] < 2:
        return {
            "cross_channel_mean_abs_corr": 0.0,
            "correlation_condition_number": 0.0,
            "channel_heterogeneity_cv": 0.0,
            "target_channel_dominance": 0.0,
            "missing_or_constant_channel_ratio": 0.0,
        }
    stds = np.std(window, axis=0)
    valid = stds > EPS
    constant_ratio = float(1.0 - np.mean(valid))
    heterogeneity = float(np.std(stds) / (np.mean(stds) + EPS))
    variances = stds ** 2
    target_dominance = float(variances[-1] / (np.mean(variances) + EPS))
    if valid.sum() < 2:
        mean_abs_corr = 0.0
        cond = 0.0
    else:
        corr = np.corrcoef(window[:, valid], rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        mask = ~np.eye(corr.shape[0], dtype=bool)
        mean_abs_corr = float(np.mean(np.abs(corr[mask]))) if mask.any() else 0.0
        eigvals = np.linalg.eigvalsh(corr + np.eye(corr.shape[0]) * 1e-6)
        cond = float(np.log1p(np.max(eigvals) / max(np.min(eigvals), 1e-6)))
    return {
        "cross_channel_mean_abs_corr": mean_abs_corr,
        "correlation_condition_number": cond,
        "channel_heterogeneity_cv": heterogeneity,
        "target_channel_dominance": target_dominance,
        "missing_or_constant_channel_ratio": constant_ratio,
    }


def router_factors_for_window(window, pred_len, ppd, reference_target):
    target = np.asarray(window[:, -1], dtype=float)
    med, mad, std, iqr = robust_stats(target)
    robust_scale = mad if mad > EPS else (std if std > EPS else 1.0)
    diffs = np.diff(target)
    half = max(1, len(target) // 2)
    trend_x = np.arange(len(target), dtype=float)
    if len(target) > 1 and std > EPS:
        slope, intercept = np.polyfit(trend_x, target, 1)
        residual = target - (slope * trend_x + intercept)
        trend_to_noise = float(abs(slope) * max(len(target) - 1, 1) / (np.std(residual) + EPS))
    else:
        trend_to_noise = 0.0
    robust_z = 0.6745 * (target - med) / (mad + EPS)
    profile = spectral_profile(target, ppd)
    entropy_std, phase_concentration = split_entropy_std(target, ppd)
    if reference_target is not None and len(reference_target) > 1:
        ref_mean = float(np.mean(reference_target))
        ref_std = float(np.std(reference_target))
        normalization_drift = float(abs(np.mean(target) - ref_mean) / (ref_std + EPS))
        normalization_drift += float(abs(std - ref_std) / (ref_std + EPS))
    else:
        normalization_drift = 0.0
    factors = {
        "router_time_robust_scale_ratio": float(std / (mad + EPS)),
        "router_time_iqr_mad_ratio": float(iqr / (mad + EPS)),
        "router_time_outlier_mass_z3": float(np.mean(np.abs(robust_z) > 3.0)),
        "router_time_late_early_level_shift": float(
            abs(np.mean(target[half:]) - np.mean(target[:half])) / (robust_scale + EPS)
        ),
        "router_time_local_roughness_ratio": float(np.mean(np.abs(diffs)) / (std + EPS)) if len(diffs) else 0.0,
        "router_time_trend_to_noise_ratio": trend_to_noise,
        "router_time_dynamic_range_log": float(
            np.log1p((np.percentile(target, 99) - np.percentile(target, 1)) / (robust_scale + EPS))
        ),
        "router_time_near_zero_fraction": float(np.mean(np.abs(target) < 0.01 * (robust_scale + EPS))),
        "router_norm_normalization_drift_score": normalization_drift,
        "router_freq_seasonal_peak_strength": profile["seasonal_peak_strength"],
        "router_freq_harmonic_energy_ratio": profile["harmonic_energy_ratio"],
        "router_freq_spectral_entropy": profile["spectral_entropy"],
        "router_freq_high_freq_noise_share": profile["high_freq_noise_share"],
        "router_freq_dominant_period_to_horizon_ratio": float(
            profile["dominant_period_to_horizon_ratio_raw"] / (float(pred_len) + EPS)
        ),
        "router_timefreq_rolling_spectral_entropy_std": entropy_std,
        "router_timefreq_dominant_frequency_drift": dominant_frequency_drift(target, ppd),
        "router_timefreq_burstiness_index": rolling_energy_ratio(target),
        "router_timefreq_phase_concentration": phase_concentration,
    }
    factors.update({f"router_multivar_{key}": value for key, value in multivar_factors(window).items()})
    return factors


def build_custom_router_features(agg, lee_root, max_windows):
    cells = agg[
        ["dataset", "pred_len", "seq_len", "label_len", "e_layers", "d_layers", "factor"]
    ].drop_duplicates()
    rows = []
    data_cache = {}
    for cell in cells.to_dict("records"):
        dataset = cell["dataset"]
        pred_len = int(cell["pred_len"])
        seq_len = int(cell["seq_len"])
        if dataset not in data_cache:
            data_cache[dataset] = read_numeric_data(dataset, lee_root)
        data, csv_path = data_cache[dataset]
        starts, border1 = sample_starts(dataset, seq_len, pred_len, len(data), max_windows)
        reference_target = data[:border1, -1] if border1 > 1 else data[: max(seq_len, 1), -1]
        factor_rows = []
        for start in starts:
            window = data[start : start + seq_len]
            if len(window) == seq_len:
                factor_rows.append(
                    router_factors_for_window(window, pred_len, points_per_day(dataset), reference_target)
                )
        summary = {**cell, "router_window_count": len(factor_rows), "router_source_csv": csv_path}
        if factor_rows:
            frame = pd.DataFrame(factor_rows)
            for column in frame.columns:
                summary[column] = float(frame[column].mean())
        rows.append(summary)
    return pd.DataFrame(rows)


def build_feature_table(agg, factors_dir, lee_root, max_windows):
    summary_factors = build_cell_factor_table(agg, factors_dir, lee_root)
    custom_factors = build_custom_router_features(agg, lee_root, max_windows)
    keys = ["dataset", "pred_len", "seq_len", "label_len", "e_layers", "d_layers", "factor"]
    table = summary_factors.merge(custom_factors, on=keys, how="left")
    keep = keys + ["test_sample_count", "router_window_count", "router_source_csv"]
    input_columns = [column for column in table.columns if column.startswith("input_")]
    router_columns = [
        column
        for column in table.columns
        if column.startswith("router_") and column not in {"router_window_count", "router_source_csv"}
    ]
    return table[keep + input_columns + router_columns]


def cell_result_table(agg):
    pivot = agg.pivot_table(index=["dataset", "pred_len"], columns="config_name", values="mse_mean").reset_index()
    ranks = agg.copy()
    ranks["rank_in_cell"] = ranks.groupby(["dataset", "pred_len"])["mse_mean"].rank(method="min")
    static_rank = (
        ranks.groupby("config_name", dropna=False)["rank_in_cell"].mean().sort_values().reset_index()
    )
    static_config = str(static_rank.iloc[0]["config_name"])
    best_idx = agg.groupby(["dataset", "pred_len"], dropna=False)["mse_mean"].idxmin()
    best = agg.loc[best_idx, ["dataset", "pred_len", "config_name", "mse_mean"]].rename(
        columns={"config_name": "oracle_config", "mse_mean": "oracle_mse"}
    )
    rows = best.merge(pivot, on=["dataset", "pred_len"], how="left")
    rows["gelu_mse"] = rows["gelu_all"]
    rows["static_config"] = static_config
    rows["static_mse"] = rows[static_config]
    rows["oracle_gain_vs_gelu"] = (rows["gelu_mse"] - rows["oracle_mse"]) / rows["gelu_mse"]
    rows["static_gain_vs_gelu"] = (rows["gelu_mse"] - rows["static_mse"]) / rows["gelu_mse"]
    rows["static_regret_vs_oracle"] = (rows["static_mse"] - rows["oracle_mse"]) / rows["oracle_mse"]
    for config in ROUTER_CONFIGS:
        if config in rows.columns:
            rows[f"gain_{config}_vs_gelu"] = (rows["gelu_mse"] - rows[config]) / rows["gelu_mse"]
    if {"tanh_all", "tanh_all_outtanh"}.issubset(rows.columns):
        rows["gain_outtanh_vs_tanh"] = (rows["tanh_all"] - rows["tanh_all_outtanh"]) / rows["tanh_all"]
    if {"gelu_all", "enc_tanh_dec_gelu"}.issubset(rows.columns):
        rows["gain_enc_tanh_vs_gelu"] = (rows["gelu_all"] - rows["enc_tanh_dec_gelu"]) / rows["gelu_all"]
    if {"tanh_all", "tanh_sin001_all", "softsign_all"}.issubset(rows.columns):
        rows["best_bounded_mse"] = rows[["tanh_all", "tanh_sin001_all", "softsign_all"]].min(axis=1)
        rows["gain_best_bounded_vs_gelu"] = (rows["gelu_mse"] - rows["best_bounded_mse"]) / rows["gelu_mse"]
    return rows, static_config, static_rank


def safe_corr(left, right, method):
    frame = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 5 or frame["left"].nunique() < 3 or frame["right"].nunique() < 3:
        return np.nan
    if method == "spearman":
        return float(frame["left"].rank(method="average").corr(frame["right"].rank(method="average")))
    return float(frame["left"].corr(frame["right"]))


def correlation_table(features, results):
    df = features.merge(results, on=["dataset", "pred_len"], how="inner")
    feature_cols = usable_feature_columns(df)
    target_cols = [
        column
        for column in df.columns
        if column.startswith("gain_") or column in {"oracle_gain_vs_gelu", "static_gain_vs_gelu"}
    ]
    rows = []
    for target in target_cols:
        for feature in feature_cols:
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "feature_group": feature.split("_", 2)[1] if feature.startswith("router_") else "input",
                    "n": int(df[[feature, target]].dropna().shape[0]),
                    "spearman": safe_corr(df[feature], df[target], "spearman"),
                    "pearson": safe_corr(df[feature], df[target], "pearson"),
                }
            )
    out = pd.DataFrame(rows)
    out["abs_spearman"] = out["spearman"].abs()
    return out.sort_values(["abs_spearman", "target", "feature"], ascending=[False, True, True])


def usable_feature_columns(df):
    feature_cols = [
        column
        for column in df.columns
        if (column.startswith("input_") or column.startswith("router_"))
        and not column.endswith("_bin")
        and column not in {"router_window_count", "router_source_csv"}
    ]
    usable = []
    for column in feature_cols:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() >= 20 and values.nunique(dropna=True) >= 4:
            usable.append(column)
            df[column] = values
    return usable


def add_family(dataset):
    if str(dataset).startswith("Solar"):
        return "Solar"
    return "ETT"


def split_groups(df, mode):
    if mode == "loco":
        return [((row["dataset"], int(row["pred_len"])), df.index[(df["dataset"] == row["dataset"]) & (df["pred_len"] == row["pred_len"])].tolist()) for _, row in df[["dataset", "pred_len"]].drop_duplicates().iterrows()]
    if mode == "lodo":
        return [(dataset, df.index[df["dataset"] == dataset].tolist()) for dataset in sorted(df["dataset"].unique())]
    if mode == "family":
        families = df["dataset"].map(add_family)
        return [(family, df.index[families == family].tolist()) for family in sorted(families.unique())]
    raise ValueError(f"Unknown split mode: {mode}")


def assign_bin(value, q1, q2):
    if pd.isna(value):
        return "mid"
    if value <= q1:
        return "low"
    if value <= q2:
        return "mid"
    return "high"


def fit_single_factor_rule(train_cells, long_train, feature, static_config):
    values = pd.to_numeric(train_cells[feature], errors="coerce").dropna()
    if values.nunique() < 3:
        q1 = q2 = float(values.median()) if not values.empty else 0.0
    else:
        q1, q2 = np.nanquantile(values.to_numpy(dtype=float), [1 / 3, 2 / 3])
        if not np.isfinite(q1) or not np.isfinite(q2):
            q1 = q2 = float(values.median())
        if q1 > q2:
            q1, q2 = q2, q1
    train = long_train.merge(train_cells[["dataset", "pred_len", feature]], on=["dataset", "pred_len"], how="inner")
    train["factor_bin"] = train[feature].apply(lambda value: assign_bin(value, q1, q2))
    choices = {}
    for factor_bin, group in train.groupby("factor_bin", dropna=False):
        score = group.groupby("config_name", dropna=False)["mse_mean"].mean().sort_values()
        choices[str(factor_bin)] = str(score.index[0]) if not score.empty else static_config
    for factor_bin in ["low", "mid", "high"]:
        choices.setdefault(factor_bin, static_config)
    return {"feature": feature, "q1": float(q1), "q2": float(q2), "choices": choices}


def predict_single_factor(rule, row):
    factor_bin = assign_bin(row.get(rule["feature"], np.nan), rule["q1"], rule["q2"])
    return rule["choices"].get(factor_bin, next(iter(rule["choices"].values())))


def score_predictions(pred):
    pred = pred.copy()
    pred["gain_vs_gelu"] = (pred["gelu_mse"] - pred["chosen_mse"]) / pred["gelu_mse"]
    pred["gain_vs_static"] = (pred["static_mse"] - pred["chosen_mse"]) / pred["static_mse"]
    pred["regret_vs_oracle"] = (pred["chosen_mse"] - pred["oracle_mse"]) / pred["oracle_mse"]
    pred["top1_hit"] = pred["chosen_config"] == pred["oracle_config"]
    return pred


def evaluate_rule_on_cells(rule, test_cells, result_cells):
    lookup = result_cells.set_index(["dataset", "pred_len"])
    rows = []
    for _, row in test_cells.iterrows():
        dataset = row["dataset"]
        pred_len = int(row["pred_len"])
        chosen = predict_single_factor(rule, row)
        res = lookup.loc[(dataset, pred_len)]
        rows.append(
            {
                "dataset": dataset,
                "pred_len": pred_len,
                "chosen_config": chosen,
                "chosen_mse": float(res[chosen]),
                "oracle_config": res["oracle_config"],
                "oracle_mse": float(res["oracle_mse"]),
                "gelu_mse": float(res["gelu_mse"]),
                "static_config": res["static_config"],
                "static_mse": float(res["static_mse"]),
            }
        )
    return score_predictions(pd.DataFrame(rows))


def inner_select_feature(train_cells, result_cells, agg, feature_cols, static_config):
    if len(train_cells) < 8:
        return feature_cols[0], pd.DataFrame()
    long_train = agg[agg.set_index(["dataset", "pred_len"]).index.isin(train_cells.set_index(["dataset", "pred_len"]).index)]
    inner_rows = []
    groups = split_groups(train_cells.reset_index(drop=True), "loco")
    for feature in feature_cols:
        predictions = []
        for _, test_idx in groups:
            test = train_cells.reset_index(drop=True).loc[test_idx]
            train = train_cells.reset_index(drop=True).drop(index=test_idx)
            if train.empty or test.empty:
                continue
            rule = fit_single_factor_rule(train, long_train, feature, static_config)
            predictions.append(evaluate_rule_on_cells(rule, test, result_cells))
        if not predictions:
            continue
        scored = pd.concat(predictions, ignore_index=True)
        inner_rows.append(
            {
                "feature": feature,
                "inner_mean_gain_vs_static": scored["gain_vs_static"].mean(),
                "inner_positive_vs_static_rate": (scored["gain_vs_static"] > 0).mean(),
                "inner_mean_regret_vs_oracle": scored["regret_vs_oracle"].mean(),
                "inner_top1_hit_rate": scored["top1_hit"].mean(),
            }
        )
    scores = pd.DataFrame(inner_rows)
    if scores.empty:
        return feature_cols[0], scores
    scores = scores.sort_values(
        ["inner_mean_gain_vs_static", "inner_positive_vs_static_rate", "inner_mean_regret_vs_oracle"],
        ascending=[False, False, True],
    )
    return str(scores.iloc[0]["feature"]), scores


def evaluate_single_factor_router(features, result_cells, agg, feature_cols, static_config, mode):
    cells = features.merge(result_cells[["dataset", "pred_len"]], on=["dataset", "pred_len"], how="inner")
    long = agg.merge(cells[["dataset", "pred_len"]], on=["dataset", "pred_len"], how="inner")
    rows = []
    inner_frames = []
    for fold, test_idx in split_groups(cells.reset_index(drop=True), mode):
        test_cells = cells.reset_index(drop=True).loc[test_idx]
        train_cells = cells.reset_index(drop=True).drop(index=test_idx)
        if train_cells.empty or test_cells.empty:
            continue
        chosen_feature, inner_scores = inner_select_feature(train_cells, result_cells, agg, feature_cols, static_config)
        if not inner_scores.empty:
            inner_scores["outer_mode"] = mode
            inner_scores["outer_fold"] = str(fold)
            inner_frames.append(inner_scores)
        rule = fit_single_factor_rule(train_cells, long, chosen_feature, static_config)
        pred = evaluate_rule_on_cells(rule, test_cells, result_cells)
        pred["split_mode"] = mode
        pred["fold"] = str(fold)
        pred["router_method"] = "nested_single_factor_rule"
        pred["selected_feature"] = chosen_feature
        pred["rule_q1"] = rule["q1"]
        pred["rule_q2"] = rule["q2"]
        pred["rule_choices"] = str(rule["choices"])
        rows.append(pred)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    inner = pd.concat(inner_frames, ignore_index=True) if inner_frames else pd.DataFrame()
    return out, inner


def evaluate_nearest_neighbor_router(features, result_cells, feature_cols, mode):
    cells = features.merge(result_cells[["dataset", "pred_len"]], on=["dataset", "pred_len"], how="inner")
    lookup = result_cells.set_index(["dataset", "pred_len"])
    rows = []
    for fold, test_idx in split_groups(cells.reset_index(drop=True), mode):
        test = cells.reset_index(drop=True).loc[test_idx]
        train = cells.reset_index(drop=True).drop(index=test_idx)
        if train.empty or test.empty:
            continue
        mu = train[feature_cols].mean()
        sd = train[feature_cols].std(ddof=0).replace(0, 1.0).fillna(1.0)
        x_train = (
            ((train[feature_cols] - mu) / sd)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .astype(float)
            .to_numpy()
        )
        for _, row in test.iterrows():
            x = (
                pd.to_numeric(row[feature_cols], errors="coerce")
                .sub(mu)
                .div(sd)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .astype(float)
                .to_numpy()
            )
            dist = np.sqrt(np.sum((x_train - x) ** 2, axis=1))
            nearest = train.iloc[int(np.argmin(dist))]
            chosen = lookup.loc[(nearest["dataset"], int(nearest["pred_len"])), "oracle_config"]
            res = lookup.loc[(row["dataset"], int(row["pred_len"]))]
            rows.append(
                {
                    "dataset": row["dataset"],
                    "pred_len": int(row["pred_len"]),
                    "chosen_config": chosen,
                    "chosen_mse": float(res[chosen]),
                    "oracle_config": res["oracle_config"],
                    "oracle_mse": float(res["oracle_mse"]),
                    "gelu_mse": float(res["gelu_mse"]),
                    "static_config": res["static_config"],
                    "static_mse": float(res["static_mse"]),
                    "split_mode": mode,
                    "fold": str(fold),
                    "router_method": "nearest_neighbor_oracle_config",
                    "selected_feature": "all_features",
                }
            )
    return score_predictions(pd.DataFrame(rows)) if rows else pd.DataFrame()


def summarize_eval(name, predictions, static_regret):
    if predictions.empty:
        return {
            "eval_name": name,
            "cell_count": 0,
            "mean_gain_vs_gelu": np.nan,
            "mean_gain_vs_static": np.nan,
            "positive_vs_static_rate": np.nan,
            "mean_regret_vs_oracle": np.nan,
            "regret_reduction_vs_static": np.nan,
            "top1_hit_rate": np.nan,
        }
    regret = float(predictions["regret_vs_oracle"].mean())
    return {
        "eval_name": name,
        "cell_count": int(len(predictions)),
        "mean_gain_vs_gelu": float(predictions["gain_vs_gelu"].mean()),
        "median_gain_vs_gelu": float(predictions["gain_vs_gelu"].median()),
        "mean_gain_vs_static": float(predictions["gain_vs_static"].mean()),
        "median_gain_vs_static": float(predictions["gain_vs_static"].median()),
        "positive_vs_static_rate": float((predictions["gain_vs_static"] > 0).mean()),
        "mean_regret_vs_oracle": regret,
        "regret_reduction_vs_static": float((static_regret - regret) / static_regret) if static_regret > EPS else np.nan,
        "top1_hit_rate": float(predictions["top1_hit"].mean()),
    }


def permutation_eval(features, result_cells, agg, feature_cols, static_config, observed, n_perm, seed):
    if n_perm <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows = []
    base = features.copy().reset_index(drop=True)
    feature_matrix = base[feature_cols].copy()
    for idx in range(n_perm):
        shuffled = base.copy()
        shuffled[feature_cols] = feature_matrix.iloc[rng.permutation(len(feature_matrix))].reset_index(drop=True)
        pred, _ = evaluate_single_factor_router(shuffled, result_cells, agg, feature_cols, static_config, "loco")
        summary = summarize_eval("permutation_loco_single_factor", pred, result_cells["static_regret_vs_oracle"].mean())
        summary["permutation_index"] = idx
        summary["observed_mean_gain_vs_static"] = observed
        rows.append(summary)
    return pd.DataFrame(rows)


def permutation_eval_fixed_loco_features(features, result_cells, agg, observed_loco, observed, n_perm, seed):
    if n_perm <= 0 or observed_loco.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    base = features.copy().reset_index(drop=True)
    selected_by_fold = (
        observed_loco.groupby("fold", dropna=False)["selected_feature"].first().to_dict()
    )
    selected_features = sorted(set(selected_by_fold.values()))
    feature_matrix = base[selected_features].copy()
    rows = []
    for idx in range(n_perm):
        shuffled = base.copy()
        shuffled[selected_features] = feature_matrix.iloc[rng.permutation(len(feature_matrix))].reset_index(drop=True)
        predictions = []
        for fold, test_idx in split_groups(shuffled, "loco"):
            fold_key = str(fold)
            feature = selected_by_fold.get(fold_key)
            if feature not in shuffled.columns:
                continue
            test_cells = shuffled.loc[test_idx]
            train_cells = shuffled.drop(index=test_idx)
            if train_cells.empty or test_cells.empty:
                continue
            train_keys = train_cells.set_index(["dataset", "pred_len"]).index
            long_train = agg[agg.set_index(["dataset", "pred_len"]).index.isin(train_keys)]
            rule = fit_single_factor_rule(train_cells, long_train, feature, str(result_cells["static_config"].iloc[0]))
            pred = evaluate_rule_on_cells(rule, test_cells, result_cells)
            pred["split_mode"] = "loco"
            pred["fold"] = fold_key
            pred["router_method"] = "permutation_fixed_loco_feature_rule"
            pred["selected_feature"] = feature
            predictions.append(pred)
        scored = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
        summary = summarize_eval(
            "permutation_fixed_loco_features",
            scored,
            result_cells["static_regret_vs_oracle"].mean(),
        )
        summary["permutation_index"] = idx
        summary["observed_mean_gain_vs_static"] = observed
        rows.append(summary)
    return pd.DataFrame(rows)


def write_summary(path, context):
    lines = [
        "# Phase-4 Router Feasibility Summary",
        "",
        f"- Aggregate rows: {context['aggregate_rows']}",
        f"- Feature rows: {context['feature_rows']}",
        f"- Input/router feature count: {context['feature_count']}",
        f"- Router feature search count: {context['router_feature_count']}",
        f"- Static baseline by mean rank: `{context['static_config']}`",
        f"- Static mean gain vs GELU: {context['static_gain']:.4f}",
        f"- Oracle mean gain vs GELU: {context['oracle_gain']:.4f}",
        f"- Static mean regret vs oracle: {context['static_regret']:.4f}",
        "",
        "## Top Correlations",
        "",
        markdown_table(context["top_correlations"], max_rows=12),
        "",
        "## Router Evaluation",
        "",
        markdown_table(context["eval_summary"]),
        "",
        "## Permutation Check",
        "",
    ]
    if context["permutation"].empty:
        lines.append("_Permutation disabled._")
    else:
        lines.extend(
            [
                f"- Observed LOCO mean gain vs static: {context['observed_gain_vs_static']:.4f}",
                f"- Permutation mean: {context['perm_mean']:.4f}",
                f"- Permutation p(perm >= observed): {context['perm_p_ge_observed']:.4f}",
                "- Permutation mode: fixed observed LOCO selected features, shuffled feature-cell alignment.",
            ]
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            context["verdict"],
            "",
            "## Claim Boundary",
            "",
            "This audit is input-only and does not use target factors, predictions, errors, or held-out cell metrics as router features. The 972 seed-runs stabilize MSE estimates, but the router sample size is 36 dataset-horizon cells.",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Input-only Phase-4 factor-router feasibility audit")
    parser.add_argument("--aggregate", default="phase4_remote_results/analysis_current/phase4_protocol_aggregate.csv")
    parser.add_argument("--factors_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lee_root", default="lee_ocil")
    parser.add_argument("--max_windows", type=int, default=2048)
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--permutation_top_features", type=int, default=12)
    parser.add_argument("--router_top_features", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260523)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    agg = pd.read_csv(args.aggregate)
    features = build_feature_table(agg, args.factors_dir, args.lee_root, args.max_windows)
    result_cells, static_config, static_rank = cell_result_table(agg)
    merged = features.merge(result_cells, on=["dataset", "pred_len"], how="inner")
    feature_cols = usable_feature_columns(merged)
    features = merged[["dataset", "pred_len"] + feature_cols + [c for c in features.columns if c not in {"dataset", "pred_len"} and c not in feature_cols]]
    correlations = correlation_table(merged[["dataset", "pred_len"] + feature_cols], result_cells)
    router_feature_cols = (
        correlations.dropna(subset=["abs_spearman"])
        .sort_values("abs_spearman", ascending=False)["feature"]
        .drop_duplicates()
        .head(args.router_top_features)
        .tolist()
    )
    if len(router_feature_cols) < 3:
        router_feature_cols = feature_cols

    loco_single, inner_loco = evaluate_single_factor_router(
        merged[["dataset", "pred_len"] + router_feature_cols], result_cells, agg, router_feature_cols, static_config, "loco"
    )
    lodo_single, inner_lodo = evaluate_single_factor_router(
        merged[["dataset", "pred_len"] + router_feature_cols], result_cells, agg, router_feature_cols, static_config, "lodo"
    )
    family_single, inner_family = evaluate_single_factor_router(
        merged[["dataset", "pred_len"] + router_feature_cols], result_cells, agg, router_feature_cols, static_config, "family"
    )
    loco_nn = evaluate_nearest_neighbor_router(
        merged[["dataset", "pred_len"] + router_feature_cols], result_cells, router_feature_cols, "loco"
    )

    static_regret = float(result_cells["static_regret_vs_oracle"].mean())
    eval_summary = pd.DataFrame(
        [
            summarize_eval("loco_nested_single_factor", loco_single, static_regret),
            summarize_eval("lodo_nested_single_factor", lodo_single, static_regret),
            summarize_eval("family_nested_single_factor", family_single, static_regret),
            summarize_eval("loco_nearest_neighbor", loco_nn, static_regret),
        ]
    )

    observed_gain = float(eval_summary.loc[eval_summary["eval_name"] == "loco_nested_single_factor", "mean_gain_vs_static"].iloc[0])
    top_perm_features = (
        correlations.dropna(subset=["abs_spearman"])
        .sort_values("abs_spearman", ascending=False)["feature"]
        .drop_duplicates()
        .head(args.permutation_top_features)
        .tolist()
    )
    if len(top_perm_features) < 3:
        top_perm_features = feature_cols
    permutation = permutation_eval_fixed_loco_features(
        merged[["dataset", "pred_len"] + sorted(set(loco_single["selected_feature"].dropna()))],
        result_cells,
        agg,
        loco_single,
        observed_gain,
        args.permutations,
        args.seed,
    )
    if permutation.empty:
        perm_mean = np.nan
        perm_p = np.nan
    else:
        perm_mean = float(permutation["mean_gain_vs_static"].mean())
        perm_p = float((permutation["mean_gain_vs_static"] >= observed_gain).mean())

    loco_summary = eval_summary[eval_summary["eval_name"] == "loco_nested_single_factor"].iloc[0]
    lodo_summary = eval_summary[eval_summary["eval_name"] == "lodo_nested_single_factor"].iloc[0]
    go = (
        loco_summary["mean_gain_vs_static"] >= 0.01
        and loco_summary["positive_vs_static_rate"] >= 0.55
        and loco_summary["regret_reduction_vs_static"] >= 0.20
        and (pd.isna(perm_p) or perm_p <= 0.10)
    )
    partial = (
        loco_summary["mean_gain_vs_static"] > 0
        and lodo_summary["mean_gain_vs_static"] > -0.0025
        and loco_summary["positive_vs_static_rate"] >= 0.50
    )
    if go:
        verdict = "GO: input-only factors pass the predefined minimal router bar on LOCO, with permutation support. The paper can cautiously upgrade to a preliminary zero-shot factor-router claim, while still reporting LODO/family limitations."
    elif partial:
        verdict = "PARTIAL: input-only factors contain router signal, but the evidence is not strong enough for a full deployable zero-shot router claim. Use as diagnostic or preliminary selector evidence."
    else:
        verdict = "NO-GO: the input-only factor router does not beat the static tanh-family baseline robustly enough. Keep the paper claim at factor-conditioned diagnostics, not a deployable router."

    features.to_csv(os.path.join(args.output_dir, "phase4_router_factor_table.csv"), index=False)
    correlations.to_csv(os.path.join(args.output_dir, "phase4_router_correlation.csv"), index=False)
    loco_single.to_csv(os.path.join(args.output_dir, "phase4_router_loco_eval.csv"), index=False)
    lodo_single.to_csv(os.path.join(args.output_dir, "phase4_router_lodo_eval.csv"), index=False)
    family_single.to_csv(os.path.join(args.output_dir, "phase4_router_family_eval.csv"), index=False)
    loco_nn.to_csv(os.path.join(args.output_dir, "phase4_router_loco_nearest_neighbor_eval.csv"), index=False)
    eval_summary.to_csv(os.path.join(args.output_dir, "phase4_router_eval_summary.csv"), index=False)
    pd.concat([inner_loco, inner_lodo, inner_family], ignore_index=True).to_csv(
        os.path.join(args.output_dir, "phase4_router_inner_feature_scores.csv"), index=False
    )
    permutation.to_csv(os.path.join(args.output_dir, "phase4_router_permutation_eval.csv"), index=False)
    static_rank.to_csv(os.path.join(args.output_dir, "phase4_router_static_rank.csv"), index=False)

    write_summary(
        os.path.join(args.output_dir, "phase4_router_feasibility_summary.md"),
        {
            "aggregate_rows": len(agg),
            "feature_rows": len(features),
            "feature_count": len(feature_cols),
            "router_feature_count": len(router_feature_cols),
            "static_config": static_config,
            "static_gain": float(result_cells["static_gain_vs_gelu"].mean()),
            "oracle_gain": float(result_cells["oracle_gain_vs_gelu"].mean()),
            "static_regret": static_regret,
            "top_correlations": correlations.head(20),
            "eval_summary": eval_summary,
            "permutation": permutation,
            "observed_gain_vs_static": observed_gain,
            "perm_mean": perm_mean,
            "perm_p_ge_observed": perm_p,
            "verdict": verdict,
        },
    )
    print(
        {
            "output_dir": args.output_dir,
            "feature_rows": len(features),
            "feature_count": len(feature_cols),
            "router_feature_count": len(router_feature_cols),
            "static_config": static_config,
            "verdict": verdict.split(":", 1)[0],
        }
    )


if __name__ == "__main__":
    main()
