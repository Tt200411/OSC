#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from phase4_model_aware_router import build_model_cells, config_action, standardize_feature_frame
from phase4_report_utils import markdown_table


DEFAULTS = {
    "d_model": 512,
    "n_heads": 8,
    "d_ff": 2048,
    "batch_size": 32,
    "train_epochs": 6,
    "patch_len": 0,
    "stride": 0,
    "model_scope": "Informer-fullgrid",
    "model_family": "Informer",
    "baseline_config": "gelu_all",
    "search_space": "phase5_newdataset_oracle",
    "placement": "",
}

CONFIGS = [
    "gelu_all",
    "tanh_all",
    "softsign_all",
    "tanh_sin001_all",
    "enc_tanh_dec_gelu",
    "enc_gelu_dec_tanh",
    "tanh_all_outtanh",
]

ACTIONS = {
    "regime_tanh_vs_gelu": "gain_tanh_all_vs_gelu",
    "regime_softsign_vs_gelu": "gain_softsign_all_vs_gelu",
    "regime_tanh_sin_vs_gelu": "gain_tanh_sin001_all_vs_gelu",
    "placement_encoder_tanh_vs_gelu": "gain_enc_tanh_dec_gelu_vs_gelu",
    "placement_decoder_tanh_vs_gelu": "gain_enc_gelu_dec_tanh_vs_gelu",
    "placement_output_tanh_vs_tanh": "gain_outtanh_vs_tanh",
    "placement_all_tanh_vs_best_split": "gain_alltanh_vs_best_split",
    "oracle_gain_vs_gelu": "oracle_gain_vs_gelu",
}

PROFILE_FEATURES = [
    "input_volatility",
    "input_mean_abs_change",
    "input_lag1_autocorr",
    "input_spectral_entropy",
    "input_high_freq_energy_ratio",
    "input_low_freq_energy_ratio",
    "router_time_local_roughness_ratio",
    "router_time_trend_to_noise_ratio",
    "router_freq_spectral_entropy",
    "router_freq_high_freq_noise_share",
    "router_freq_seasonal_peak_strength",
    "router_timefreq_burstiness_index",
    "router_multivar_cross_channel_mean_abs_corr",
    "router_norm_normalization_drift_score",
    "pred_to_seq_ratio",
]


def ensure_defaults(frame):
    out = frame.copy()
    for column, value in DEFAULTS.items():
        if column not in out.columns:
            out[column] = value
        out[column] = out[column].fillna(value)
    numeric = [
        "pred_len",
        "seq_len",
        "label_len",
        "e_layers",
        "d_layers",
        "factor",
        "d_model",
        "n_heads",
        "d_ff",
        "batch_size",
        "train_epochs",
        "patch_len",
        "stride",
    ]
    for column in numeric:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    out["action"] = out.apply(lambda row: config_action(row["config_name"], row["baseline_config"]), axis=1)
    return out


def build_wide(agg):
    index_cols = ["dataset", "pred_len", "seq_len", "label_len", "e_layers", "d_layers", "factor"]
    wide = agg.pivot_table(index=index_cols, columns="config_name", values="mse_mean").reset_index()
    for config in CONFIGS:
        if config not in wide.columns:
            wide[config] = np.nan
    for config in CONFIGS[1:]:
        wide[f"gain_{config}_vs_gelu"] = (wide["gelu_all"] - wide[config]) / wide["gelu_all"]
    wide["gain_outtanh_vs_tanh"] = (wide["tanh_all"] - wide["tanh_all_outtanh"]) / wide["tanh_all"]
    split_best = wide[["enc_tanh_dec_gelu", "enc_gelu_dec_tanh"]].min(axis=1)
    wide["gain_alltanh_vs_best_split"] = (split_best - wide["tanh_all"]) / split_best
    wide["oracle_config"] = wide[CONFIGS].idxmin(axis=1)
    wide["oracle_mse"] = wide[CONFIGS].min(axis=1)
    wide["oracle_gain_vs_gelu"] = (wide["gelu_all"] - wide["oracle_mse"]) / wide["gelu_all"]
    return wide


def feature_columns(frame):
    columns = [
        column
        for column in frame.columns
        if (column.startswith("input_") or column.startswith("router_"))
        and column not in {"router_window_count", "router_source_csv"}
        and not column.endswith("_bin")
    ]
    columns += ["pred_len", "pred_to_seq_ratio"]
    usable = []
    for column in dict.fromkeys(columns):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() >= 7 and values.nunique(dropna=True) >= 3:
            frame[column] = values
            usable.append(column)
    return usable


def safe_spearman(left, right):
    data = pd.DataFrame({"left": left, "right": right}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 5 or data["left"].nunique() < 3 or data["right"].nunique() < 3:
        return np.nan
    return float(data["left"].rank(method="average").corr(data["right"].rank(method="average")))


def action_summary(frame):
    rows = []
    for action, target in ACTIONS.items():
        values = pd.to_numeric(frame[target], errors="coerce")
        rows.append(
            {
                "action": action,
                "target": target,
                "n": int(values.notna().sum()),
                "mean_uplift": float(values.mean()),
                "median_uplift": float(values.median()),
                "positive_rate": float((values > 0).mean()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows).sort_values("action")


def correlation_table(frame, features):
    rows = []
    for action, target in ACTIONS.items():
        for feature in features:
            corr = safe_spearman(frame[feature], frame[target])
            if not np.isnan(corr):
                rows.append(
                    {
                        "action": action,
                        "target": target,
                        "feature": feature,
                        "spearman": corr,
                        "abs_spearman": abs(corr),
                    }
                )
    return pd.DataFrame(rows).sort_values(["action", "abs_spearman"], ascending=[True, False])


def fit_best_threshold(frame, target, features, min_support=2):
    best = None
    y = pd.to_numeric(frame[target], errors="coerce").to_numpy(float)
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        clean = values.dropna()
        if clean.nunique() < 3:
            continue
        thresholds = sorted(set(np.nanquantile(clean, np.linspace(0.15, 0.85, 8))))
        for threshold in thresholds:
            for direction in ["high", "low"]:
                fire = values >= threshold if direction == "high" else values <= threshold
                fire = fire.fillna(False).to_numpy(bool)
                if int(fire.sum()) < min_support:
                    continue
                fired = y[fire]
                if len(fired) == 0 or float(np.nanmean(fired)) <= 0:
                    continue
                candidate = {
                    "feature": feature,
                    "direction": direction,
                    "threshold": float(threshold),
                    "fired_count": int(fire.sum()),
                    "fired_rate": float(fire.mean()),
                    "precision": float(np.nanmean(fired > 0)),
                    "mean_uplift_if_fired": float(np.nanmean(fired)),
                    "policy_mean_uplift": float(np.nanmean(np.where(fire, y, 0.0))),
                }
                key = (candidate["policy_mean_uplift"], candidate["precision"], candidate["mean_uplift_if_fired"])
                if best is None or key > (best["policy_mean_uplift"], best["precision"], best["mean_uplift_if_fired"]):
                    best = candidate
    if best is None:
        return {
            "feature": "abstain",
            "direction": "none",
            "threshold": np.nan,
            "fired_count": 0,
            "fired_rate": 0.0,
            "precision": np.nan,
            "mean_uplift_if_fired": np.nan,
            "policy_mean_uplift": 0.0,
        }
    return best


def apply_rule(frame, rule):
    if rule["feature"] == "abstain":
        return np.zeros(len(frame), dtype=bool)
    values = pd.to_numeric(frame[rule["feature"]], errors="coerce")
    fire = values >= rule["threshold"] if rule["direction"] == "high" else values <= rule["threshold"]
    return fire.fillna(False).to_numpy(bool)


def threshold_tables(frame, features):
    rule_rows = []
    cv_rows = []
    for action, target in ACTIONS.items():
        rule = fit_best_threshold(frame, target, features)
        rule_rows.append({"action": action, "target": target, **rule})
        predictions = []
        for dataset in sorted(frame["dataset"].astype(str).unique()):
            train = frame[frame["dataset"].astype(str) != dataset].copy()
            test = frame[frame["dataset"].astype(str) == dataset].copy()
            fold_rule = fit_best_threshold(train, target, features)
            fire = apply_rule(test, fold_rule)
            for idx, row in enumerate(test.reset_index(drop=True).to_dict("records")):
                uplift = float(row[target])
                predictions.append(
                    {
                        "action": action,
                        "fold": dataset,
                        "dataset": row["dataset"],
                        "pred_len": int(row["pred_len"]),
                        "uplift": uplift,
                        "fire": bool(fire[idx]),
                        "policy_uplift": uplift if fire[idx] else 0.0,
                        "rule_feature": fold_rule["feature"],
                    }
                )
        pred = pd.DataFrame(predictions)
        fired = pred[pred["fire"]]
        cv_rows.append(
            {
                "action": action,
                "target": target,
                "mean_policy_uplift": float(pred["policy_uplift"].mean()) if not pred.empty else np.nan,
                "fire_rate": float(pred["fire"].mean()) if not pred.empty else np.nan,
                "precision_if_fire": float((fired["uplift"] > 0).mean()) if not fired.empty else np.nan,
                "always_fire_mean_uplift": float(frame[target].mean()),
                "positive_rate": float((frame[target] > 0).mean()),
                "features_used": ";".join(pred["rule_feature"].value_counts().head(3).index.astype(str))
                if not pred.empty
                else "",
            }
        )
    return pd.DataFrame(rule_rows), pd.DataFrame(cv_rows)


def input_zero_audit(frame):
    input_cols = [c for c in frame.columns if c.startswith("input_") and not c.endswith("_bin")]
    rows = []
    for dataset, group in frame.groupby("dataset", dropna=False):
        numeric = group[input_cols].apply(pd.to_numeric, errors="coerce")
        rows.append(
            {
                "dataset": dataset,
                "input_feature_count": len(input_cols),
                "nonzero_input_features": int(((numeric.abs().sum(axis=0) > 1e-12)).sum()),
                "zero_or_missing_input_feature_rate": float(((numeric.abs().sum(axis=0) <= 1e-12) | numeric.isna().all(axis=0)).mean()),
            }
        )
    return pd.DataFrame(rows)


def write_report(path, context):
    input_zero = context["input_zero_audit"]
    max_zero_rate = (
        float(input_zero["zero_or_missing_input_feature_rate"].max()) if not input_zero.empty else np.nan
    )
    if pd.notna(max_zero_rate) and max_zero_rate <= 0.05:
        input_sanity_note = (
            "- input_* factors pass the granularity sanity check for this audit; no dataset has material "
            "zero/missing collapse."
        )
    else:
        input_sanity_note = (
            "- Some input_* factors still show zero/missing collapse; prefer router_* recomputed window "
            "features until the factor extraction is fixed for those datasets."
        )
    lines = [
        "# Phase-5 Factor Signal Audit",
        "",
        "Scope: Weather, Exchange, and ILI 3-seed oracle matrix. The audit checks whether input/statistical/time-frequency factors are aligned with activation regime and placement uplift.",
        "",
        "## Verdict",
        "",
        context["verdict"],
        "",
        "## Action Uplift Summary",
        "",
        markdown_table(context["action_summary"], max_rows=20),
        "",
        "## Leave-Dataset Threshold Check",
        "",
        markdown_table(context["leave_dataset_rules"], max_rows=20),
        "",
        "## Router-Only Leave-Dataset Threshold Check",
        "",
        markdown_table(context["router_only_leave_dataset_rules"], max_rows=20),
        "",
        "## Top Spearman Correlations",
        "",
    ]
    top_corr = (
        context["correlations"].sort_values(["action", "abs_spearman"], ascending=[True, False])
        .groupby("action", dropna=False)
        .head(4)
        .reset_index(drop=True)
    )
    lines.extend([markdown_table(top_corr, max_rows=40), ""])
    lines.extend(
        [
            "## Best In-Sample Single-Feature Rules",
            "",
            markdown_table(context["threshold_rules"].sort_values("policy_mean_uplift", ascending=False), max_rows=20),
            "",
            "## Best Router-Only In-Sample Single-Feature Rules",
            "",
            markdown_table(
                context["router_only_threshold_rules"].sort_values("policy_mean_uplift", ascending=False),
                max_rows=20,
            ),
            "",
            "## Factor Extraction Sanity",
            "",
            markdown_table(context["input_zero_audit"], max_rows=10),
            "",
            "## Cell Profile",
            "",
            markdown_table(context["profile"], max_rows=12),
            "",
            "## Interpretation Boundary",
            "",
            "- The sample size is 9 dataset-horizon cells, so high in-sample correlations are hypothesis-generating only.",
            "- The current signal supports coarse abstain/calibration gates better than exact zero-shot config selection.",
            input_sanity_note,
            "- Router-only leave-dataset checks are the more conservative read of generalization because they remove dependence on the input_* extraction path.",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Audit Phase-5 factor/action signal on new datasets.")
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--factors_dir", required=True)
    parser.add_argument("--lee_root", default="lee_ocil")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    agg = ensure_defaults(pd.read_csv(args.aggregate))
    wide = build_wide(agg)
    cells, _ = build_model_cells(agg)
    features = standardize_feature_frame(cells, args.factors_dir, args.lee_root)
    drop = [column for column in ["oracle_config", "oracle_mse"] if column in features.columns]
    merged = wide.merge(features.drop(columns=drop), on=["dataset", "pred_len", "seq_len", "label_len", "e_layers", "d_layers", "factor"], how="left")
    if "pred_to_seq_ratio" not in merged.columns:
        merged["pred_to_seq_ratio"] = merged["pred_len"] / merged["seq_len"].replace(0, np.nan)

    features_used = feature_columns(merged)
    summary = action_summary(merged)
    corr = correlation_table(merged, features_used)
    threshold_rules, leave_dataset_rules = threshold_tables(merged, features_used)
    router_only_features = [
        feature for feature in features_used if feature.startswith("router_") or feature in {"pred_len", "pred_to_seq_ratio"}
    ]
    router_only_threshold_rules, router_only_leave_dataset_rules = threshold_tables(merged, router_only_features)
    zero_audit = input_zero_audit(merged)
    profile_cols = ["dataset", "pred_len", "oracle_config", "oracle_gain_vs_gelu"] + [
        column for column in PROFILE_FEATURES if column in merged.columns
    ]
    profile = merged[profile_cols].sort_values(["dataset", "pred_len"])

    verdict = (
        "There is interpretable signal, but it is not yet strong enough for a deployable exact zero-shot router. "
        "The strongest reproducible pattern is placement-level: decoder-side tanh is broadly safe/positive except one Weather-96 failure, "
        "while static all-layer bounded regimes and output tanh need abstain gates. "
        "Leave-dataset threshold rules only remain positive for decoder-side placement and oracle gain, so the current factors are better "
        "as calibration/abstain features than as a standalone exact regime+placement selector."
    )

    outputs = {
        "phase5_factor_signal_cells.csv": merged,
        "phase5_factor_signal_action_summary.csv": summary,
        "phase5_factor_signal_top_correlations.csv": corr,
        "phase5_factor_signal_threshold_rules.csv": threshold_rules,
        "phase5_factor_signal_leave_dataset_rules.csv": leave_dataset_rules,
        "phase5_factor_signal_router_only_threshold_rules.csv": router_only_threshold_rules,
        "phase5_factor_signal_router_only_leave_dataset_rules.csv": router_only_leave_dataset_rules,
        "phase5_factor_signal_input_zero_audit.csv": zero_audit,
        "phase5_factor_signal_profile.csv": profile,
    }
    for filename, frame in outputs.items():
        frame.to_csv(os.path.join(args.output_dir, filename), index=False)

    report_path = os.path.join(args.output_dir, "phase5_factor_signal_audit.md")
    write_report(
        report_path,
        {
            "verdict": verdict,
            "action_summary": summary,
            "correlations": corr,
            "threshold_rules": threshold_rules,
            "leave_dataset_rules": leave_dataset_rules,
            "router_only_threshold_rules": router_only_threshold_rules,
            "router_only_leave_dataset_rules": router_only_leave_dataset_rules,
            "input_zero_audit": zero_audit,
            "profile": profile,
        },
    )
    print(report_path)


if __name__ == "__main__":
    main()
