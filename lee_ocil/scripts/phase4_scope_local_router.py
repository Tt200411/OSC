#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from phase4_model_aware_router import (
    build_model_cells,
    config_action,
    load_scope,
    split_groups,
    standardize_feature_frame,
    usable_features,
)
from phase4_model_aware_router_ablation import FREQ_TOKENS
from phase4_report_utils import markdown_table


CELL_KEYS = ["model_scope", "dataset", "pred_len"]


def bin_value(value, q1, q2):
    if pd.isna(value):
        return "mid"
    if value <= q1:
        return "low"
    if value <= q2:
        return "mid"
    return "high"


def load_all(args):
    scopes = [
        load_scope(args.informer, "Informer-fullgrid", "Informer", "gelu_all", "fullgrid"),
        load_scope(args.patchtst, "PatchTST-transfer", "PatchTST", "gelu_ffn_linear", "transfer"),
        load_scope(args.itransformer_stage1, "iTransformer-stage1", "iTransformer", "gelu_ffn_linear", "ffn_only"),
        load_scope(args.itransformer_placeprobe, "iTransformer-placeprobe", "iTransformer", "gelu_ffn_linear", "placement_probe"),
    ]
    results = pd.concat(scopes, ignore_index=True)
    results["action"] = results.apply(lambda row: config_action(row["config_name"], row["baseline_config"]), axis=1)
    cells, results_with_actions = build_model_cells(results)
    cells = standardize_feature_frame(cells, args.factors_dir, args.lee_root)
    score_cols = CELL_KEYS + ["oracle_mse", "baseline_mse", "model_family", "baseline_config"]
    scored = results_with_actions.merge(cells[score_cols], on=CELL_KEYS, how="left")
    scored["config_regret_vs_oracle"] = (scored["mse_mean"] - scored["oracle_mse"]) / scored["oracle_mse"]
    scored["config_gain_vs_baseline"] = (scored["baseline_mse"] - scored["mse_mean"]) / scored["baseline_mse"]
    return cells, scored


def feature_sets(cells, top_k):
    all_cols = usable_features(cells)
    data_cols = [c for c in all_cols if c.startswith(("input_", "router_"))]
    freq_cols = [c for c in data_cols if any(token in c for token in FREQ_TOKENS)]
    stat_cols = [c for c in data_cols if c not in set(freq_cols)]
    # Keep deterministic, compact candidate lists. Scope-local feature selection
    # is intentionally conservative because PatchTST/iTransformer have few cells.
    return {
        "data_only": data_cols[:top_k],
        "frequency_only": freq_cols[:top_k],
        "statistical_only": stat_cols[:top_k],
    }


def keys_frame(df):
    return df[CELL_KEYS].drop_duplicates()


def subset_results_by_cells(results, cells):
    return results.merge(keys_frame(cells), on=CELL_KEYS, how="inner")


def cell_lookup_rows(results, row):
    mask = (
        (results["model_scope"] == row["model_scope"])
        & (results["dataset"] == row["dataset"])
        & (results["pred_len"] == row["pred_len"])
    )
    return results[mask]


def choose_scope_static(train_results, scope):
    scope_rows = train_results[train_results["model_scope"] == scope]
    if scope_rows.empty:
        return None
    means = scope_rows.groupby("config_name", dropna=False)["config_regret_vs_oracle"].mean().sort_values()
    return str(means.index[0])


def choose_scope_action_config(train_results, scope, action):
    rows = train_results[(train_results["model_scope"] == scope) & (train_results["action"] == action)]
    if rows.empty:
        return choose_scope_static(train_results, scope)
    means = rows.groupby("config_name", dropna=False)["config_regret_vs_oracle"].mean().sort_values()
    return str(means.index[0])


def best_test_row_for_config(results, row, config):
    candidates = cell_lookup_rows(results, row)
    selected = candidates[candidates["config_name"] == config]
    if selected.empty:
        baseline = str(row["baseline_config"])
        selected = candidates[candidates["config_name"] == baseline]
    if selected.empty:
        selected = candidates.sort_values("mse_mean").head(1)
    return selected.iloc[0]


def fit_feature_rule(train_cells, train_results, feature, target_kind, margin=0.0, min_support=2):
    values = pd.to_numeric(train_cells[feature], errors="coerce")
    if values.nunique(dropna=True) < 3:
        q1 = q2 = float(values.median()) if values.notna().any() else 0.0
    else:
        q1, q2 = np.nanquantile(values.to_numpy(float), [1 / 3, 2 / 3])
    cells_for_merge = train_cells[CELL_KEYS + [feature]].copy()
    cells_for_merge["feature_bin"] = cells_for_merge[feature].apply(lambda v: bin_value(v, q1, q2))
    merged = train_results.merge(cells_for_merge[CELL_KEYS + ["feature_bin"]], on=CELL_KEYS, how="inner")
    choices = {}
    for (scope, feature_bin), group in merged.groupby(["model_scope", "feature_bin"], dropna=False):
        if target_kind == "action":
            scores = group.groupby("action", dropna=False)["config_regret_vs_oracle"].mean().sort_values()
            best = str(scores.index[0]) if not scores.empty else "baseline"
        else:
            scores = group.groupby("config_name", dropna=False)["config_regret_vs_oracle"].mean().sort_values()
            best = str(scores.index[0]) if not scores.empty else None
        static_config = choose_scope_static(train_results, scope)
        static_rows = group[group["config_name"] == static_config]
        best_rows = group[group["action"] == best] if target_kind == "action" else group[group["config_name"] == best]
        static_score = float(static_rows["config_regret_vs_oracle"].mean()) if not static_rows.empty else np.inf
        best_score = float(best_rows["config_regret_vs_oracle"].mean()) if not best_rows.empty else np.inf
        support = int(best_rows[CELL_KEYS].drop_duplicates().shape[0]) if not best_rows.empty else 0
        use_dynamic = support >= min_support and best_score + margin < static_score
        choices[(str(scope), str(feature_bin))] = {
            "choice": best,
            "static_config": static_config,
            "support": support,
            "train_best_score": best_score,
            "train_static_score": static_score,
            "use_dynamic": use_dynamic,
        }
    return {"feature": feature, "q1": float(q1), "q2": float(q2), "choices": choices, "target_kind": target_kind}


def predict_rule(rule, row, train_results, test_results, abstain):
    scope = str(row["model_scope"])
    feature_bin = bin_value(row.get(rule["feature"], np.nan), rule["q1"], rule["q2"])
    entry = rule["choices"].get((scope, str(feature_bin)))
    static_config = choose_scope_static(train_results, scope)
    used_dynamic = False
    if entry is None:
        config = static_config
        choice = "static_fallback"
    elif abstain and not entry["use_dynamic"]:
        config = entry["static_config"] or static_config
        choice = "abstain_static"
    else:
        choice = entry["choice"]
        if rule["target_kind"] == "action":
            config = choose_scope_action_config(train_results, scope, choice)
        else:
            config = choice
        used_dynamic = True
    selected = best_test_row_for_config(test_results, row, config)
    return {
        "chosen_config": str(selected["config_name"]),
        "chosen_action": str(selected["action"]),
        "chosen_mse": float(selected["mse_mean"]),
        "selected_feature": rule["feature"],
        "choice": choice,
        "used_dynamic": used_dynamic,
        "feature_bin": str(feature_bin),
    }


def evaluate_predictions(pred):
    out = pred.copy()
    out["gain_vs_baseline"] = (out["baseline_mse"] - out["chosen_mse"]) / out["baseline_mse"]
    out["gain_vs_train_static"] = (out["train_static_mse"] - out["chosen_mse"]) / out["train_static_mse"]
    out["regret_vs_oracle"] = (out["chosen_mse"] - out["oracle_mse"]) / out["oracle_mse"]
    out["config_hit"] = out["chosen_config"] == out["oracle_config"]
    out["action_hit"] = out["chosen_action"] == out["oracle_action"]
    return out


def score_rule_on_train(train_cells, train_results, feature, target_kind, abstain):
    rule = fit_feature_rule(train_cells, train_results, feature, target_kind)
    rows = []
    for _, row in train_cells.iterrows():
        static_config = choose_scope_static(train_results, row["model_scope"])
        static_row = best_test_row_for_config(train_results, row, static_config)
        pred = predict_rule(rule, row, train_results, train_results, abstain)
        rows.append(
            {
                **pred,
                "baseline_mse": row["baseline_mse"],
                "oracle_mse": row["oracle_mse"],
                "oracle_config": row["oracle_config"],
                "oracle_action": row["oracle_action"],
                "train_static_config": str(static_row["config_name"]),
                "train_static_mse": float(static_row["mse_mean"]),
            }
        )
    scored = evaluate_predictions(pd.DataFrame(rows))
    return scored["gain_vs_train_static"].mean(), scored["regret_vs_oracle"].mean(), rule


def fit_scope_local_rules(train_cells, train_results, features, target_kind, abstain):
    rules = {}
    selection_rows = []
    for scope, scope_cells in train_cells.groupby("model_scope", dropna=False):
        scope_results = train_results[train_results["model_scope"] == scope]
        if len(scope_cells) < 4 or not features:
            continue
        best = None
        for feature in features:
            gain, regret, rule = score_rule_on_train(scope_cells, scope_results, feature, target_kind, abstain)
            row = {
                "model_scope": scope,
                "target_kind": target_kind,
                "abstain": abstain,
                "feature": feature,
                "train_gain_vs_static": gain,
                "train_regret_vs_oracle": regret,
            }
            selection_rows.append(row)
            key = (gain, -regret)
            if best is None or key > best[0]:
                best = (key, rule)
        if best is not None:
            rules[str(scope)] = best[1]
    return rules, pd.DataFrame(selection_rows)


def evaluate_variant(cells, results, features, mode, feature_set, target_kind, abstain):
    predictions = []
    selections = []
    reset = cells.reset_index(drop=True)
    for fold, test_idx in split_groups(reset, mode):
        test_cells = reset.loc[test_idx]
        train_cells = reset.drop(index=test_idx)
        train_results = subset_results_by_cells(results, train_cells)
        test_results = subset_results_by_cells(results, test_cells)
        rules, selection = fit_scope_local_rules(train_cells, train_results, features, target_kind, abstain)
        if not selection.empty:
            selection["fold"] = str(fold)
            selection["eval_name"] = mode
            selection["feature_set"] = feature_set
            selections.append(selection)
        for _, row in test_cells.iterrows():
            static_config = choose_scope_static(train_results, row["model_scope"])
            static_row = best_test_row_for_config(test_results, row, static_config)
            rule = rules.get(str(row["model_scope"]))
            if rule is None:
                pred = {
                    "chosen_config": str(static_row["config_name"]),
                    "chosen_action": str(static_row["action"]),
                    "chosen_mse": float(static_row["mse_mean"]),
                    "selected_feature": "scope_static",
                    "choice": "scope_static",
                    "used_dynamic": False,
                    "feature_bin": "",
                }
            else:
                pred = predict_rule(rule, row, train_results, test_results, abstain)
            predictions.append(
                {
                    **pred,
                    "eval_name": mode,
                    "fold": str(fold),
                    "feature_set": feature_set,
                    "target_kind": target_kind,
                    "abstain": abstain,
                    "model_scope": row["model_scope"],
                    "model_family": row["model_family"],
                    "dataset": row["dataset"],
                    "pred_len": row["pred_len"],
                    "baseline_mse": row["baseline_mse"],
                    "oracle_mse": row["oracle_mse"],
                    "oracle_config": row["oracle_config"],
                    "oracle_action": row["oracle_action"],
                    "train_static_config": str(static_row["config_name"]),
                    "train_static_mse": float(static_row["mse_mean"]),
                }
            )
    return evaluate_predictions(pd.DataFrame(predictions)), pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()


def summarize(pred):
    return {
        "cell_count": int(len(pred)),
        "mean_gain_vs_baseline": float(pred["gain_vs_baseline"].mean()),
        "mean_gain_vs_train_static": float(pred["gain_vs_train_static"].mean()),
        "positive_vs_train_static_rate": float((pred["gain_vs_train_static"] > 0).mean()),
        "mean_regret_vs_oracle": float(pred["regret_vs_oracle"].mean()),
        "config_hit_rate": float(pred["config_hit"].mean()),
        "action_hit_rate": float(pred["action_hit"].mean()),
        "dynamic_use_rate": float(pred["used_dynamic"].mean()),
    }


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    cells, results = load_all(args)
    sets = feature_sets(cells, args.top_k)
    summary_rows = []
    all_preds = []
    all_selections = []
    requested_feature_sets = set(args.feature_sets) if args.feature_sets else None
    requested_target_kinds = set(args.target_kinds) if args.target_kinds else {"config", "action"}
    requested_abstain = set(args.abstain_modes) if args.abstain_modes else {"direct", "abstain"}
    for mode in args.modes:
        for feature_set, features in sets.items():
            if requested_feature_sets is not None and feature_set not in requested_feature_sets:
                continue
            if not features:
                continue
            for target_kind in ["config", "action"]:
                if target_kind not in requested_target_kinds:
                    continue
                for abstain in [False, True]:
                    abstain_name = "abstain" if abstain else "direct"
                    if abstain_name not in requested_abstain:
                        continue
                    pred, selection = evaluate_variant(cells, results, features, mode, feature_set, target_kind, abstain)
                    name = f"{feature_set}:{target_kind}:{'abstain' if abstain else 'direct'}"
                    pred["variant"] = name
                    summary = summarize(pred)
                    summary.update(
                        {
                            "eval_name": mode,
                            "feature_set": feature_set,
                            "target_kind": target_kind,
                            "abstain": abstain,
                            "variant": name,
                            "top_features": ";".join(features[:8]),
                        }
                    )
                    summary_rows.append(summary)
                    all_preds.append(pred)
                    if not selection.empty:
                        selection["variant"] = name
                        all_selections.append(selection)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["eval_name", "mean_gain_vs_train_static", "positive_vs_train_static_rate"],
        ascending=[True, False, False],
    )
    pred_all = pd.concat(all_preds, ignore_index=True)
    summary.to_csv(os.path.join(args.output_dir, "phase4_scope_local_router_summary.csv"), index=False)
    pred_all.to_csv(os.path.join(args.output_dir, "phase4_scope_local_router_predictions.csv"), index=False)
    if all_selections:
        pd.concat(all_selections, ignore_index=True).to_csv(
            os.path.join(args.output_dir, "phase4_scope_local_router_feature_selection.csv"),
            index=False,
        )
    lines = [
        "# Scope-Local Two-Stage Router Summary",
        "",
        f"- Cells: {len(cells)}",
        f"- Top-k feature candidates per set: {args.top_k}",
        f"- Evaluation modes: {', '.join(args.modes)}",
        "",
        "This protocol first gates by model scope, then learns a data-factor",
        "refinement rule inside each seen scope. The static comparator is also",
        "selected from the training fold for the same scope.",
        "",
        markdown_table(summary),
    ]
    with open(os.path.join(args.output_dir, "phase4_scope_local_router_summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print({"output_dir": args.output_dir, "rows": len(summary), "cells": len(cells)})


def main():
    parser = argparse.ArgumentParser(description="Scope-local two-stage activation router")
    parser.add_argument("--factors_dir", required=True)
    parser.add_argument("--output_dir", default="phase4_remote_results/analysis_current/factors_router/scope_local")
    parser.add_argument("--lee_root", default="lee_ocil")
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--modes", nargs="+", default=["leave_cell", "leave_dataset"])
    parser.add_argument("--feature_sets", nargs="*", default=None)
    parser.add_argument("--target_kinds", nargs="*", default=None)
    parser.add_argument("--abstain_modes", nargs="*", default=None)
    parser.add_argument("--informer", default="phase4_remote_results/analysis_current/phase4_protocol_aggregate.csv")
    parser.add_argument("--patchtst", default="patchtst_remote_results/analysis_patchtst_transfer_actfix/patchtst_transfer_aggregate.csv")
    parser.add_argument("--itransformer_stage1", default="itransformer_remote_results/analysis_itransformer_stage1_actfix3/itransformer_protocol_aggregate.csv")
    parser.add_argument("--itransformer_placeprobe", default="itransformer_remote_results/analysis_itransformer_placeprobe_20260522/itransformer_protocol_aggregate.csv")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
