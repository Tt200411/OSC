#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from phase4_model_aware_router import build_model_cells, config_action, load_scope, standardize_feature_frame
from phase4_pairwise_factor_router import (
    ACTION_CONFIG_CANDIDATES,
    CELL_KEYS,
    CORE_ACTIONS,
    action_to_exact_config,
    apply_rule,
    cell_id_frame,
    feature_groups,
    fit_action_rule,
    make_pair_rows,
    rank_static_configs,
)
from phase4_report_utils import markdown_table


ACTION_FAMILY = {
    "regime_tanh_vs_default": "regime",
    "regime_tanh_sin_vs_default": "regime",
    "regime_softsign_vs_default": "regime",
    "regime_best_bounded_vs_default": "regime",
    "placement_encoder_tanh_vs_default": "placement",
    "placement_decoder_tanh_vs_default": "placement",
    "placement_all_tanh_vs_best_split": "placement",
    "placement_output_tanh_vs_best_linear": "placement",
    "placement_output_tanh_matched": "placement",
}


def load_informer_results(path):
    results = load_scope(path, "Informer-fullgrid", "Informer", "gelu_all", "fullgrid")
    results["action"] = results.apply(lambda row: config_action(row["config_name"], row["baseline_config"]), axis=1)
    return results


def new_cells_from_matrix(matrix_path):
    matrix = pd.read_csv(matrix_path)
    rows = []
    for _, row in matrix.drop_duplicates(["dataset", "pred_len"]).iterrows():
        rows.append(
            {
                "model_scope": "Informer-fullgrid",
                "model_family": "Informer",
                "search_space": "phase5_newdataset_oracle",
                "dataset": row["dataset"],
                "pred_len": int(row["pred_len"]),
                "seq_len": int(row["seq_len"]),
                "label_len": int(row["label_len"]),
                "e_layers": int(row["e_layers"]),
                "d_layers": int(row["d_layers"]),
                "factor": int(row["factor"]),
                "d_model": 512,
                "n_heads": 8,
                "d_ff": 2048,
                "batch_size": int(row.get("batch_size", 32)),
                "train_epochs": int(row.get("train_epochs", 6)),
                "patch_len": 0,
                "stride": 0,
                "baseline_config": "gelu_all",
                "available_configs": tuple(sorted(matrix[matrix["dataset"].eq(row["dataset"]) & matrix["pred_len"].eq(row["pred_len"])]["config_name"].astype(str).unique())),
            }
        )
    return pd.DataFrame(rows)


def build_new_action_rows(new_cells, actions):
    rows = []
    for _, cell in new_cells.iterrows():
        available = set(cell["available_configs"])
        for action_name in actions:
            candidates = ACTION_CONFIG_CANDIDATES.get(action_name, [])
            action_config = next((candidate for candidate in candidates if candidate in available), "")
            if action_name == "placement_output_tanh_vs_best_linear" and "tanh_all_outtanh" in available:
                action_config = "tanh_all_outtanh"
            if action_name == "placement_output_tanh_matched" and "tanh_all_outtanh" in available:
                action_config = "tanh_all_outtanh"
            if action_name == "regime_best_bounded_vs_default":
                action_config = "tanh_sin001_all" if "tanh_sin001_all" in available else ""
            if not action_config:
                continue
            rows.append(
                {
                    **{key: cell[key] for key in CELL_KEYS},
                    "model_family": cell["model_family"],
                    "action_name": action_name,
                    "action_family": ACTION_FAMILY.get(action_name, "regime"),
                    "action_config": action_config,
                    "comparator_config": "gelu_all",
                }
            )
    return pd.DataFrame(rows)


def fit_rules(train_pairs, feature_sets, args):
    rule_rows = []
    for feature_set, features in feature_sets.items():
        if not features:
            continue
        for action_name, action_df in train_pairs.groupby("action_name", dropna=False):
            if action_name not in CORE_ACTIONS or len(action_df) < args.min_pairs:
                continue
            rule = fit_action_rule(action_df.reset_index(drop=True), features, args.margin, args.min_support, args.min_precision)
            rule_rows.append(
                {
                    "feature_set": feature_set,
                    "action_name": action_name,
                    "action_family": str(action_df["action_family"].iloc[0]),
                    **rule,
                }
            )
    return pd.DataFrame(rule_rows)


def apply_rules(new_pairs, rules):
    rows = []
    for _, rule in rules.iterrows():
        action_rows = new_pairs[
            (new_pairs["action_name"].astype(str) == str(rule["action_name"]))
        ].copy()
        if action_rows.empty:
            continue
        fire = apply_rule(action_rows, rule)
        pred = action_rows[CELL_KEYS + ["model_family", "action_name", "action_family", "action_config", "comparator_config"]].copy()
        pred["feature_set"] = rule["feature_set"]
        pred["recommend"] = fire
        pred["selected_feature"] = rule["feature"]
        pred["rule_direction"] = rule["direction"]
        pred["rule_threshold"] = rule["threshold"]
        pred["rule_objective"] = rule["objective"]
        pred["rule_precision"] = rule["precision"]
        pred["rule_mean_uplift_if_fired"] = rule["mean_uplift_if_fired"]
        rows.append(pred)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def choose_configs(predictions, new_cells, train_results, fallback_policies):
    train_cells = cell_id_frame(build_model_cells(train_results)[0])
    static_by_scope = rank_static_configs(train_results, train_cells)
    rows = []
    for feature_set, group in predictions.groupby("feature_set", dropna=False):
        for _, cell in new_cells.iterrows():
            cell_group = group[
                (group["model_scope"].astype(str) == str(cell["model_scope"]))
                & (group["dataset"].astype(str) == str(cell["dataset"]))
                & (group["pred_len"].astype(int) == int(cell["pred_len"]))
            ]
            recs = cell_group[cell_group["recommend"]].copy()
            available = set(cell["available_configs"])
            candidates = []
            for _, rec in recs.iterrows():
                config = action_to_exact_config(
                    str(rec["action_name"]),
                    available,
                    train_results,
                    train_cells,
                    "Informer-fullgrid",
                )
                if not config:
                    continue
                candidates.append(
                    {
                        "chosen_config": config,
                        "chosen_action_name": str(rec["action_name"]),
                        "chosen_action_family": str(rec["action_family"]),
                        "rule_objective": float(rec.get("rule_objective", 0.0)),
                        "rule_precision": float(rec.get("rule_precision", np.nan)),
                        "selected_feature": str(rec.get("selected_feature", "")),
                    }
                )
            selected = None
            if candidates:
                selected = sorted(
                    candidates,
                    key=lambda item: (
                        item["rule_objective"],
                        -1.0 if pd.isna(item["rule_precision"]) else item["rule_precision"],
                        item["chosen_config"],
                    ),
                    reverse=True,
                )[0]
            for fallback_policy in fallback_policies:
                if selected:
                    chosen = selected.copy()
                    abstained = False
                else:
                    fallback_config = "gelu_all"
                    if fallback_policy == "train_static":
                        fallback_config = static_by_scope.get("Informer-fullgrid", "gelu_all")
                        if fallback_config not in available:
                            fallback_config = "gelu_all"
                    chosen = {
                        "chosen_config": fallback_config,
                        "chosen_action_name": "abstain",
                        "chosen_action_family": "abstain",
                        "rule_objective": 0.0,
                        "rule_precision": np.nan,
                        "selected_feature": "abstain",
                    }
                    abstained = True
                rows.append(
                    {
                        "feature_set": feature_set,
                        "fallback_policy": fallback_policy,
                        "model_scope": cell["model_scope"],
                        "dataset": cell["dataset"],
                        "pred_len": int(cell["pred_len"]),
                        "seq_len": int(cell["seq_len"]),
                        "label_len": int(cell["label_len"]),
                        "e_layers": int(cell["e_layers"]),
                        "d_layers": int(cell["d_layers"]),
                        "factor": int(cell["factor"]),
                        "baseline_config": "gelu_all",
                        "train_static_config": static_by_scope.get("Informer-fullgrid", "gelu_all"),
                        "available_configs": ";".join(cell["available_configs"]),
                        "recommendation_count": len(candidates),
                        "abstained": abstained,
                        **chosen,
                    }
                )
    return pd.DataFrame(rows)


def summarize_choices(choices):
    if choices.empty:
        return pd.DataFrame()
    return (
        choices.groupby(["feature_set", "fallback_policy"], dropna=False)
        .agg(
            cell_count=("dataset", "size"),
            abstain_rate=("abstained", "mean"),
            mean_recommendation_count=("recommendation_count", "mean"),
            top_chosen_configs=("chosen_config", lambda s: ";".join(s.value_counts().head(5).index.astype(str))),
            top_actions=("chosen_action_name", lambda s: ";".join(s.value_counts().head(5).index.astype(str))),
            top_features=("selected_feature", lambda s: ";".join(s.value_counts().head(5).index.astype(str))),
        )
        .reset_index()
        .sort_values(["feature_set", "fallback_policy"])
    )


def write_report(path, choices, summary, rules):
    lines = [
        "# Phase-5 Zero-Shot Router Predictions",
        "",
        "These predictions are trained only on the completed Informer full grid and use input statistical/time-frequency features for new datasets. No new-dataset oracle MSE is used here.",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Cell Predictions",
        "",
        markdown_table(choices),
        "",
        "## Learned Rules",
        "",
        markdown_table(rules),
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    train_results = load_informer_results(args.informer)
    train_cells, _ = build_model_cells(train_results)
    train_cells = standardize_feature_frame(train_cells, args.factors_dir, args.lee_root)
    train_pairs = make_pair_rows(train_results)
    train_pairs = train_pairs.merge(
        train_cells,
        on=CELL_KEYS + ["model_family", "search_space", "baseline_config"],
        how="left",
    )
    train_pairs = train_pairs[train_pairs["action_name"].isin(CORE_ACTIONS)].copy()
    feature_sets = feature_groups(train_pairs, args.top_k, include_topology=args.include_topology)
    if args.feature_sets:
        keep = set(args.feature_sets)
        feature_sets = {name: cols for name, cols in feature_sets.items() if name in keep}
    rules = fit_rules(train_pairs, feature_sets, args)

    new_cells = new_cells_from_matrix(args.matrix)
    new_cells = standardize_feature_frame(new_cells, args.factors_dir, args.lee_root)
    new_pairs = build_new_action_rows(new_cells, sorted(CORE_ACTIONS))
    feature_cols = sorted(
        {
            column
            for columns in feature_sets.values()
            for column in columns
            if column in new_cells.columns and column not in set(CELL_KEYS)
        }
    )
    new_pairs = new_pairs.merge(new_cells[CELL_KEYS + feature_cols], on=CELL_KEYS, how="left")
    predictions = apply_rules(new_pairs, rules)
    choices = choose_configs(
        predictions,
        new_cells,
        train_results,
        [item.strip() for item in args.fallback_policies.split(",") if item.strip()],
    )
    summary = summarize_choices(choices)

    rules.to_csv(os.path.join(args.output_dir, "phase5_zero_shot_rules.csv"), index=False)
    predictions.to_csv(os.path.join(args.output_dir, "phase5_zero_shot_pair_predictions.csv"), index=False)
    choices.to_csv(os.path.join(args.output_dir, "phase5_zero_shot_cell_predictions.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, "phase5_zero_shot_summary.csv"), index=False)
    write_report(os.path.join(args.output_dir, "phase5_zero_shot_report.md"), choices, summary, rules)
    print(
        {
            "output_dir": args.output_dir,
            "rules": len(rules),
            "pair_predictions": len(predictions),
            "cell_predictions": len(choices),
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Zero-shot Phase-5 router predictions for new datasets")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--factors_dir", required=True)
    parser.add_argument("--output_dir", default=".aris/phase5_zero_shot_router")
    parser.add_argument("--lee_root", default="lee_ocil")
    parser.add_argument("--informer", default="phase4_remote_results/analysis_current/phase4_protocol_aggregate.csv")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--margin", type=float, default=0.002)
    parser.add_argument("--min_support", type=int, default=4)
    parser.add_argument("--min_precision", type=float, default=0.55)
    parser.add_argument("--min_pairs", type=int, default=10)
    parser.add_argument("--fallback_policies", default="gelu,train_static")
    parser.add_argument("--feature_sets", nargs="+", default=["all_input", "data", "statistical", "frequency"])
    parser.add_argument("--include_topology", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
