#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from phase4_model_aware_router import build_model_cells, load_scope, standardize_feature_frame
from phase4_pairwise_factor_router import (
    CELL_KEYS,
    CORE_ACTIONS,
    combine_pair_predictions_to_cells,
    feature_groups,
    fit_action_rule,
    make_pair_rows,
    apply_rule,
)
from phase4_report_utils import markdown_table


NEW_DATASETS = {"Weather", "Exchange", "ILI"}


def load_old_informer(path):
    old = load_scope(path, "Informer-fullgrid", "Informer", "gelu_all", "fullgrid")
    old["action"] = old.apply(lambda row: _config_action(row["config_name"], row["baseline_config"]), axis=1)
    return old


def _config_action(config_name, baseline_config):
    name = str(config_name)
    if name == str(baseline_config):
        return "baseline"
    if "outtanh" in name or "out=tanh" in name:
        return "output_tanh"
    if "tanh" in name or "softsign" in name:
        return "bounded_linear"
    return "other_linear"


def load_new_oracle(path):
    frame = pd.read_csv(path)
    defaults = {
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
    }
    for column, value in defaults.items():
        frame[column] = value
    frame["action"] = frame.apply(lambda row: _config_action(row["config_name"], row["baseline_config"]), axis=1)
    return frame


def is_new_cell(df):
    return df["dataset"].astype(str).isin(NEW_DATASETS)


def same_cell_mask(df, cells):
    markers = cells[CELL_KEYS].drop_duplicates().assign(__holdout=True)
    merged = df[CELL_KEYS].merge(markers, on=CELL_KEYS, how="left")
    return merged["__holdout"].eq(True).to_numpy(bool)


def split_new_cells(cells, mode):
    new_cells = cells[is_new_cell(cells)].copy()
    if mode == "supplement_leave_new_dataset":
        return [
            (str(dataset), group[CELL_KEYS].drop_duplicates())
            for dataset, group in new_cells.groupby("dataset", dropna=False)
        ]
    if mode == "supplement_leave_new_horizon":
        return [
            (str(int(pred_len)), group[CELL_KEYS].drop_duplicates())
            for pred_len, group in new_cells.groupby("pred_len", dropna=False)
        ]
    if mode == "supplement_leave_new_cell":
        folds = []
        for _, row in new_cells[CELL_KEYS].drop_duplicates().sort_values(CELL_KEYS).iterrows():
            folds.append(
                (
                    f"{row['dataset']}/{int(row['pred_len'])}",
                    pd.DataFrame([{key: row[key] for key in CELL_KEYS}]),
                )
            )
        return folds
    raise ValueError(mode)


def evaluate_supplemented(pairs, cells, feature_sets, args):
    prediction_rows = []
    rule_rows = []
    pairs = pairs[pairs["action_name"].isin(CORE_ACTIONS)].reset_index(drop=True)
    for mode in args.eval_modes:
        for fold, holdout_cells in split_new_cells(cells, mode):
            test_mask = same_cell_mask(pairs, holdout_cells)
            for feature_set, features in feature_sets.items():
                if not features:
                    continue
                for action_name, action_df in pairs.groupby("action_name", dropna=False):
                    train = action_df.loc[~test_mask[action_df.index]].reset_index(drop=True)
                    test = action_df.loc[test_mask[action_df.index]].reset_index(drop=True)
                    if test.empty or len(train) < args.min_pairs:
                        continue
                    rule = fit_action_rule(train, features, args.margin, args.min_support, args.min_precision)
                    fire = apply_rule(test, rule)
                    pred = test[
                        CELL_KEYS
                        + [
                            "model_family",
                            "action_name",
                            "action_family",
                            "action_config",
                            "comparator_config",
                            "uplift",
                        ]
                    ].copy()
                    pred["feature_set"] = feature_set
                    pred["eval_name"] = mode
                    pred["fold"] = str(fold)
                    pred["recommend"] = fire
                    pred["policy_uplift"] = np.where(fire, pred["uplift"], 0.0)
                    pred["selected_feature"] = rule["feature"]
                    pred["rule_direction"] = rule["direction"]
                    pred["rule_threshold"] = rule["threshold"]
                    pred["rule_objective"] = rule["objective"]
                    pred["rule_precision"] = rule["precision"]
                    pred["rule_mean_uplift_if_fired"] = rule["mean_uplift_if_fired"]
                    prediction_rows.append(pred)
                    rule_rows.append(
                        {
                            "eval_name": mode,
                            "fold": str(fold),
                            "feature_set": feature_set,
                            "action_name": action_name,
                            "action_family": str(test["action_family"].iloc[0]),
                            **rule,
                        }
                    )
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    rules = pd.DataFrame(rule_rows)
    return predictions, rules


def write_report(path, cell_summary, cell_predictions, rules):
    lines = [
        "# Phase-5 Oracle-Supplemented Router Evaluation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This evaluation trains router rules on the completed old Informer full grid plus the non-held-out new oracle cells, then evaluates only held-out Weather/Exchange/ILI cells.",
        "",
        "## Cell Summary",
        "",
        markdown_table(cell_summary),
        "",
        "## Cell Predictions",
        "",
        markdown_table(cell_predictions),
        "",
        "## Learned Rules",
        "",
        markdown_table(rules),
        "",
        "## Boundary",
        "",
        "- The current new oracle matrix is seed-2024 only, so this is a route-selection signal rather than final paper evidence.",
        "- If supplemented splits improve over zero-shot but remain seed-fragile, the next step is to add seeds 2025/2026 before drafting statistical claims.",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    old = load_old_informer(args.informer)
    new = load_new_oracle(args.new_oracle_aggregate)
    results = pd.concat([old, new], ignore_index=True, sort=False)
    cells, _ = build_model_cells(results)
    cells = standardize_feature_frame(cells, args.factors_dir, args.lee_root)
    pairs = make_pair_rows(results)
    feature_cols = [c for c in cells.columns if c.startswith(("input_", "router_"))]
    meta_cols = [
        "model_family",
        "search_space",
        "baseline_config",
        "d_model",
        "n_heads",
        "d_ff",
        "batch_size",
        "train_epochs",
        "patch_len",
        "stride",
        "pred_to_seq_ratio",
        "ffn_to_model_ratio",
        "has_patch",
        "patch_to_seq_ratio",
        "stride_to_patch_ratio",
        "has_decoder",
        "is_informer",
        "is_patchtst",
        "is_itransformer",
        "scope_has_output_tanh",
        "scope_config_count",
    ]
    pairs = pairs.merge(
        cells[CELL_KEYS + meta_cols + feature_cols].drop_duplicates(),
        on=CELL_KEYS + ["model_family", "search_space", "baseline_config"],
        how="left",
    )
    feature_sets = feature_groups(pairs, args.top_k, include_topology=args.include_topology)
    if args.feature_sets:
        keep = set(args.feature_sets)
        feature_sets = {name: cols for name, cols in feature_sets.items() if name in keep}
    predictions, rules = evaluate_supplemented(pairs, cells, feature_sets, args)
    cell_predictions, cell_summary = combine_pair_predictions_to_cells(
        predictions,
        pairs,
        results,
        [item.strip() for item in args.fallback_policies.split(",") if item.strip()],
        args.margin,
        True,
    )

    results.to_csv(os.path.join(args.output_dir, "phase5_calibrated_router_results.csv"), index=False)
    pairs.to_csv(os.path.join(args.output_dir, "phase5_calibrated_router_pairs.csv"), index=False)
    predictions.to_csv(os.path.join(args.output_dir, "phase5_calibrated_router_pair_predictions.csv"), index=False)
    rules.to_csv(os.path.join(args.output_dir, "phase5_calibrated_router_rules.csv"), index=False)
    cell_predictions.to_csv(os.path.join(args.output_dir, "phase5_calibrated_router_cell_predictions.csv"), index=False)
    cell_summary.to_csv(os.path.join(args.output_dir, "phase5_calibrated_router_cell_summary.csv"), index=False)
    write_report(
        os.path.join(args.output_dir, "phase5_calibrated_router_report.md"),
        cell_summary,
        cell_predictions,
        rules,
    )
    print(
        {
            "output_dir": args.output_dir,
            "result_rows": int(len(results)),
            "pair_rows": int(len(pairs)),
            "pair_predictions": int(len(predictions)),
            "cell_predictions": int(len(cell_predictions)),
            "cell_summary_rows": int(len(cell_summary)),
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate oracle-supplemented Phase-5 factor router splits")
    parser.add_argument("--informer", default="phase4_remote_results/analysis_current/phase4_protocol_aggregate.csv")
    parser.add_argument("--new_oracle_aggregate", required=True)
    parser.add_argument("--factors_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lee_root", default="lee_ocil")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--margin", type=float, default=0.002)
    parser.add_argument("--min_support", type=int, default=4)
    parser.add_argument("--min_precision", type=float, default=0.55)
    parser.add_argument("--min_pairs", type=int, default=10)
    parser.add_argument("--fallback_policies", default="gelu,train_static")
    parser.add_argument(
        "--eval_modes",
        nargs="+",
        default=[
            "supplement_leave_new_dataset",
            "supplement_leave_new_horizon",
            "supplement_leave_new_cell",
        ],
    )
    parser.add_argument("--feature_sets", nargs="+", default=["all_input", "data", "statistical", "frequency"])
    parser.add_argument("--include_topology", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
