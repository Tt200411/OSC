#!/usr/bin/env python3
import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from phase4_common import config_signature, read_all_summaries
from phase4_pairwise_factor_router import config_action
from phase4_report_utils import markdown_table


KEY_COLS = [
    "dataset",
    "pred_len",
    "seq_len",
    "label_len",
    "e_layers",
    "d_layers",
    "factor",
    "batch_size",
    "train_epochs",
    "config_name",
    "seed",
]

CELL_COLS = [
    "dataset",
    "pred_len",
    "seq_len",
    "label_len",
    "e_layers",
    "d_layers",
    "factor",
]

STATIC_CONFIGS = ["tanh_all", "softsign_all", "tanh_sin001_all"]


def normalize_numeric(frame, columns):
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def load_expected(matrix_path):
    expected = pd.read_csv(matrix_path)
    expected = normalize_numeric(
        expected,
        [
            "pred_len",
            "seq_len",
            "label_len",
            "e_layers",
            "d_layers",
            "factor",
            "batch_size",
            "train_epochs",
            "seed",
        ],
    )
    expected["activation_signature"] = expected["config_name"].map(config_signature)
    return expected


def load_observed(summary_globs, expected):
    summaries = read_all_summaries(summary_globs)
    if summaries.empty:
        return summaries, expected.copy()

    numeric_cols = [
        "pred_len",
        "seq_len",
        "label_len",
        "e_layers",
        "d_layers",
        "factor",
        "batch_size",
        "train_epochs",
        "seed",
        "mse",
        "mae",
    ]
    summaries = normalize_numeric(summaries, numeric_cols)
    expected_keys = expected[KEY_COLS].drop_duplicates()
    observed = summaries.merge(expected_keys, on=KEY_COLS, how="inner").copy()
    observed = observed[observed["mse"].notna() & observed["mae"].notna() & observed["has_config"].astype(bool)]
    if observed.empty:
        missing = expected.copy()
        return observed, missing

    observed["_dedupe_order"] = np.arange(len(observed))
    observed["_has_arrays_order"] = observed["has_arrays"].astype(bool).astype(int)
    observed = (
        observed.sort_values(["_has_arrays_order", "source_summary", "_source_order", "_dedupe_order"])
        .drop_duplicates(KEY_COLS, keep="last")
        .copy()
    )
    missing = expected.merge(observed[KEY_COLS], on=KEY_COLS, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns=["_merge"])
    return observed, missing


def aggregate_observed(observed):
    if observed.empty:
        return pd.DataFrame()
    group_cols = CELL_COLS + ["config_name", "activation_signature"]
    return (
        observed.groupby(group_cols, dropna=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            seed_count=("seed", "nunique"),
            run_count=("setting", "count"),
            servers=("server_ip", lambda s: ";".join(sorted(set(str(x) for x in s if str(x))))),
        )
        .reset_index()
        .sort_values(CELL_COLS + ["mse_mean", "config_name"])
    )


def lookup_mse(agg, cell, config):
    rows = agg[
        (agg["dataset"].astype(str) == str(cell["dataset"]))
        & (agg["pred_len"].astype(int) == int(cell["pred_len"]))
        & (agg["seq_len"].astype(int) == int(cell["seq_len"]))
        & (agg["label_len"].astype(int) == int(cell["label_len"]))
        & (agg["e_layers"].astype(int) == int(cell["e_layers"]))
        & (agg["d_layers"].astype(int) == int(cell["d_layers"]))
        & (agg["factor"].astype(int) == int(cell["factor"]))
        & (agg["config_name"].astype(str) == str(config))
    ]
    if rows.empty:
        return np.nan
    return float(rows.iloc[0]["mse_mean"])


def build_cell_oracles(agg):
    if agg.empty:
        return pd.DataFrame()
    rows = []
    for key, group in agg.groupby(CELL_COLS, dropna=False):
        cell = dict(zip(CELL_COLS, key))
        oracle = group.sort_values("mse_mean").iloc[0]
        gelu_mse = lookup_mse(agg, cell, "gelu_all")
        static_rows = group[group["config_name"].isin(STATIC_CONFIGS)].sort_values("mse_mean")
        if static_rows.empty:
            best_static_config = ""
            best_static_mse = np.nan
        else:
            best_static_config = str(static_rows.iloc[0]["config_name"])
            best_static_mse = float(static_rows.iloc[0]["mse_mean"])
        rows.append(
            {
                **cell,
                "gelu_mse": gelu_mse,
                "oracle_config": str(oracle["config_name"]),
                "oracle_action": config_action(str(oracle["config_name"]), "gelu_all"),
                "oracle_mse": float(oracle["mse_mean"]),
                "oracle_seed_count": int(oracle["seed_count"]),
                "new_best_static_config": best_static_config,
                "new_best_static_mse": best_static_mse,
                "oracle_gain_vs_gelu": (gelu_mse - float(oracle["mse_mean"])) / gelu_mse
                if gelu_mse > 0
                else np.nan,
                "new_static_gain_vs_gelu": (gelu_mse - best_static_mse) / gelu_mse
                if gelu_mse > 0 and not pd.isna(best_static_mse)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(CELL_COLS)


def score_rows(choices, agg, cell_oracles, margin):
    rows = []
    oracle_lookup = {
        (str(row["dataset"]), int(row["pred_len"]), int(row["seq_len"]), int(row["label_len"]), int(row["e_layers"]), int(row["d_layers"]), int(row["factor"])): row
        for row in cell_oracles.to_dict("records")
    }
    for choice in choices.to_dict("records"):
        key = (
            str(choice["dataset"]),
            int(choice["pred_len"]),
            int(choice["seq_len"]),
            int(choice["label_len"]),
            int(choice["e_layers"]),
            int(choice["d_layers"]),
            int(choice["factor"]),
        )
        oracle = oracle_lookup.get(key)
        if oracle is None:
            continue
        chosen_mse = lookup_mse(agg, choice, choice["chosen_config"])
        train_static_mse = lookup_mse(agg, choice, choice.get("train_static_config", ""))
        gelu_mse = float(oracle["gelu_mse"])
        oracle_mse = float(oracle["oracle_mse"])
        new_static_mse = float(oracle["new_best_static_mse"])
        gain_vs_gelu = (gelu_mse - chosen_mse) / gelu_mse if gelu_mse > 0 else np.nan
        gain_vs_train_static = (
            (train_static_mse - chosen_mse) / train_static_mse
            if train_static_mse > 0 and not pd.isna(train_static_mse)
            else np.nan
        )
        gain_vs_new_static = (
            (new_static_mse - chosen_mse) / new_static_mse
            if new_static_mse > 0 and not pd.isna(new_static_mse)
            else np.nan
        )
        oracle_gain = (gelu_mse - oracle_mse) / gelu_mse if gelu_mse > 0 else np.nan
        rows.append(
            {
                **choice,
                "chosen_mse": chosen_mse,
                "gelu_mse": gelu_mse,
                "train_static_mse": train_static_mse,
                "new_best_static_config": oracle["new_best_static_config"],
                "new_best_static_mse": new_static_mse,
                "oracle_config": oracle["oracle_config"],
                "oracle_action": oracle["oracle_action"],
                "oracle_mse": oracle_mse,
                "oracle_seed_count": oracle["oracle_seed_count"],
                "gain_vs_gelu": gain_vs_gelu,
                "gain_vs_train_static": gain_vs_train_static,
                "gain_vs_new_static": gain_vs_new_static,
                "regret_vs_oracle": (chosen_mse - oracle_mse) / oracle_mse if oracle_mse > 0 else np.nan,
                "oracle_capture_ratio": gain_vs_gelu / oracle_gain
                if oracle_gain and oracle_gain > 1e-12
                else np.nan,
                "positive_vs_gelu": gain_vs_gelu > margin,
                "positive_vs_train_static": gain_vs_train_static > margin
                if not pd.isna(gain_vs_train_static)
                else False,
                "positive_vs_new_static": gain_vs_new_static > margin
                if not pd.isna(gain_vs_new_static)
                else False,
                "config_hit": str(choice["chosen_config"]) == str(oracle["oracle_config"]),
                "action_hit": config_action(str(choice["chosen_config"]), "gelu_all")
                == str(oracle["oracle_action"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_eval(scored):
    if scored.empty:
        return pd.DataFrame()
    rows = []
    for key, group in scored.groupby(["feature_set", "fallback_policy"], dropna=False):
        feature_set, fallback_policy = key
        rows.append(
            {
                "feature_set": feature_set,
                "fallback_policy": fallback_policy,
                "cell_count": int(len(group)),
                "mean_gain_vs_gelu": float(group["gain_vs_gelu"].mean()),
                "median_gain_vs_gelu": float(group["gain_vs_gelu"].median()),
                "mean_gain_vs_train_static": float(group["gain_vs_train_static"].mean()),
                "mean_gain_vs_new_static": float(group["gain_vs_new_static"].mean()),
                "positive_vs_gelu_rate": float(group["positive_vs_gelu"].mean()),
                "positive_vs_train_static_rate": float(group["positive_vs_train_static"].mean()),
                "positive_vs_new_static_rate": float(group["positive_vs_new_static"].mean()),
                "mean_regret_vs_oracle": float(group["regret_vs_oracle"].mean()),
                "median_regret_vs_oracle": float(group["regret_vs_oracle"].median()),
                "mean_oracle_capture_ratio": float(
                    group["oracle_capture_ratio"].replace([np.inf, -np.inf], np.nan).mean()
                ),
                "config_hit_rate": float(group["config_hit"].mean()),
                "action_hit_rate": float(group["action_hit"].mean()),
                "abstain_rate": float(group["abstained"].astype(bool).mean()),
                "top_chosen_configs": ";".join(group["chosen_config"].value_counts().head(5).index.astype(str)),
                "top_oracle_configs": ";".join(group["oracle_config"].value_counts().head(5).index.astype(str)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_gain_vs_train_static", "mean_regret_vs_oracle"], ascending=[False, True]
    )


def dataset_summary(scored):
    if scored.empty:
        return pd.DataFrame()
    return (
        scored.groupby(["feature_set", "fallback_policy", "dataset"], dropna=False)
        .agg(
            cell_count=("chosen_config", "count"),
            mean_gain_vs_gelu=("gain_vs_gelu", "mean"),
            mean_gain_vs_train_static=("gain_vs_train_static", "mean"),
            mean_gain_vs_new_static=("gain_vs_new_static", "mean"),
            mean_regret_vs_oracle=("regret_vs_oracle", "mean"),
            config_hit_rate=("config_hit", "mean"),
            action_hit_rate=("action_hit", "mean"),
        )
        .reset_index()
        .sort_values(["feature_set", "fallback_policy", "dataset"])
    )


def write_report(path, expected, observed, missing, agg, cell_oracles, scored, summary, by_dataset):
    lines = [
        "# Phase-5 Router Oracle Evaluation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Evidence Status",
        "",
        f"- Expected seed-runs: {len(expected)}",
        f"- Observed deduped seed-runs: {len(observed)}",
        f"- Missing seed-runs: {len(missing)}",
        f"- Aggregated dataset-horizon-config cells: {len(agg)}",
        f"- Dataset-horizon oracle cells: {len(cell_oracles)}",
        "",
    ]
    if not cell_oracles.empty:
        oracle_display = cell_oracles[
            [
                "dataset",
                "pred_len",
                "gelu_mse",
                "new_best_static_config",
                "new_best_static_mse",
                "oracle_config",
                "oracle_mse",
                "oracle_gain_vs_gelu",
            ]
        ].copy()
        lines.extend(["## Oracle Cells", "", markdown_table(oracle_display), ""])
    if not summary.empty:
        lines.extend(["## Zero-Shot Router Summary", "", markdown_table(summary), ""])
    if not by_dataset.empty:
        lines.extend(["## By Dataset", "", markdown_table(by_dataset), ""])
    if not scored.empty:
        selected_cols = [
            "feature_set",
            "fallback_policy",
            "dataset",
            "pred_len",
            "chosen_config",
            "oracle_config",
            "gain_vs_gelu",
            "gain_vs_train_static",
            "gain_vs_new_static",
            "regret_vs_oracle",
            "config_hit",
            "abstained",
        ]
        lines.extend(["## Cell-Level Router Choices", "", markdown_table(scored[selected_cols]), ""])
    lines.extend(
        [
            "## Interpretation Boundary",
            "",
            "- A one-seed oracle matrix is a triage signal only; do not write it as a statistical conclusion.",
            "- Zero-shot rows use only old Informer full-grid rules plus input factors from the new datasets; they do not train on new dataset test errors.",
            "- `new_best_static` is an evaluation oracle for the new matrix, not a deployable zero-shot baseline.",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase-5 zero-shot router against new-dataset oracle results")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--choices", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_glob", action="append", default=[])
    parser.add_argument("--margin", type=float, default=0.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    expected = load_expected(args.matrix)
    observed, missing = load_observed(args.summary_glob, expected)
    agg = aggregate_observed(observed)
    cell_oracles = build_cell_oracles(agg)
    choices = pd.read_csv(args.choices)
    choices = normalize_numeric(choices, ["pred_len", "seq_len", "label_len", "e_layers", "d_layers", "factor"])
    scored = score_rows(choices, agg, cell_oracles, args.margin)
    summary = summarize_eval(scored)
    by_dataset = dataset_summary(scored)

    expected.to_csv(os.path.join(args.output_dir, "phase5_oracle_expected.csv"), index=False)
    observed.to_csv(os.path.join(args.output_dir, "phase5_oracle_raw_dedup.csv"), index=False)
    missing.to_csv(os.path.join(args.output_dir, "phase5_oracle_missing.csv"), index=False)
    agg.to_csv(os.path.join(args.output_dir, "phase5_oracle_aggregate.csv"), index=False)
    cell_oracles.to_csv(os.path.join(args.output_dir, "phase5_oracle_cells.csv"), index=False)
    scored.to_csv(os.path.join(args.output_dir, "phase5_zero_shot_oracle_eval_cells.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, "phase5_zero_shot_oracle_eval_summary.csv"), index=False)
    by_dataset.to_csv(os.path.join(args.output_dir, "phase5_zero_shot_oracle_eval_by_dataset.csv"), index=False)
    write_report(
        os.path.join(args.output_dir, "phase5_router_oracle_evaluation.md"),
        expected,
        observed,
        missing,
        agg,
        cell_oracles,
        scored,
        summary,
        by_dataset,
    )
    print(
        {
            "expected_seed_runs": int(len(expected)),
            "observed_seed_runs": int(len(observed)),
            "missing_seed_runs": int(len(missing)),
            "aggregate_rows": int(len(agg)),
            "oracle_cells": int(len(cell_oracles)),
            "summary_rows": int(len(summary)),
            "output_dir": args.output_dir,
        }
    )


if __name__ == "__main__":
    main()
