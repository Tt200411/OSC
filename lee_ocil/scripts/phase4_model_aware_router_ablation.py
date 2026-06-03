#!/usr/bin/env python3
import argparse
import os
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from phase4_model_aware_router import (
    build_action_table,
    build_model_cells,
    correlation_table,
    evaluate_router,
    load_scope,
    standardize_feature_frame,
    summarize,
    usable_features,
)
from phase4_report_utils import markdown_table


FREQ_TOKENS = (
    "spectral",
    "freq",
    "frequency",
    "harmonic",
    "daily",
    "cycle",
    "seasonal",
    "period",
    "timefreq",
)


def load_all(args):
    scopes = [
        load_scope(args.informer, "Informer-fullgrid", "Informer", "gelu_all", "fullgrid"),
        load_scope(args.patchtst, "PatchTST-transfer", "PatchTST", "gelu_ffn_linear", "transfer"),
        load_scope(args.itransformer_stage1, "iTransformer-stage1", "iTransformer", "gelu_ffn_linear", "ffn_only"),
        load_scope(args.itransformer_placeprobe, "iTransformer-placeprobe", "iTransformer", "gelu_ffn_linear", "placement_probe"),
    ]
    results = pd.concat(scopes, ignore_index=True)
    cells, results_with_actions = build_model_cells(results)
    actions = build_action_table(results_with_actions, cells)
    cells = standardize_feature_frame(cells, args.factors_dir, args.lee_root)
    return cells, actions, results_with_actions


def valid_cols(cells, cols):
    out = []
    for col in dict.fromkeys(cols):
        if col not in cells.columns:
            continue
        values = pd.to_numeric(cells[col], errors="coerce")
        if values.notna().sum() >= 10 and values.nunique(dropna=True) >= 2:
            cells[col] = values
            out.append(col)
    return out


def top_features(cells, cols, top_k):
    if len(cols) <= top_k:
        return cols
    corr = correlation_table(cells, cols)
    return corr["feature"].drop_duplicates().head(top_k).tolist()


def feature_sets(cells, top_k):
    all_usable = usable_features(cells)
    data_cols = [c for c in all_usable if c.startswith(("input_", "router_"))]
    freq_cols = [c for c in data_cols if any(token in c for token in FREQ_TOKENS)]
    stat_cols = [c for c in data_cols if c not in set(freq_cols)]
    topology_cols = valid_cols(
        cells,
        [
            "pred_len",
            "seq_len",
            "pred_to_seq_ratio",
            "e_layers",
            "d_layers",
            "d_model",
            "n_heads",
            "d_ff",
            "ffn_to_model_ratio",
            "batch_size",
            "train_epochs",
            "patch_len",
            "stride",
            "patch_to_seq_ratio",
            "stride_to_patch_ratio",
            "has_patch",
            "has_decoder",
            "scope_has_output_tanh",
            "scope_config_count",
        ],
    )
    model_id_cols = valid_cols(cells, ["is_informer", "is_patchtst", "is_itransformer"])
    architecture_flag_cols = valid_cols(
        cells,
        [
            "is_informer",
            "is_patchtst",
            "is_itransformer",
            "has_patch",
            "has_decoder",
            "scope_has_output_tanh",
        ],
    )
    cells["router_constant"] = 0.0
    return {
        "constant_scope_rule": ["router_constant"],
        "data_only": top_features(cells, data_cols, top_k),
        "frequency_only": top_features(cells, freq_cols, top_k),
        "statistical_only": top_features(cells, stat_cols, top_k),
        "topology_only": top_features(cells, topology_cols, top_k),
        "model_id_only": top_features(cells, model_id_cols, top_k),
        "architecture_flags": top_features(cells, architecture_flag_cols, top_k),
        "all_features": top_features(cells, all_usable, top_k),
    }


def run_ablation(args):
    os.makedirs(args.output_dir, exist_ok=True)
    cells, actions, results = load_all(args)
    sets = feature_sets(cells, args.top_k)
    rows = []
    selected_rows = []
    static_regret = float(cells["static_regret_vs_oracle"].mean())
    for granularity in args.granularity:
        for set_name, cols in sets.items():
            if not cols:
                continue
            for mode in args.modes:
                pred, inner = evaluate_router(
                    cells,
                    actions,
                    results,
                    cols,
                    mode,
                    args.selection_mode,
                    granularity,
                )
                summary = summarize(mode, pred, static_regret)
                summary.update(
                    {
                        "evaluation_granularity": granularity,
                        "feature_set": set_name,
                        "feature_count": len(cols),
                        "top_features": ";".join(cols[:8]),
                        "fallback_rate": float(pred["fallback_used"].mean()) if "fallback_used" in pred else 0.0,
                    }
                )
                rows.append(summary)
                selected = pred["selected_feature"].value_counts().reset_index()
                selected.columns = ["selected_feature", "count"]
                selected["evaluation_granularity"] = granularity
                selected["feature_set"] = set_name
                selected["eval_name"] = mode
                selected_rows.append(selected)
    out = pd.DataFrame(rows)
    out = out[
        [
            "evaluation_granularity",
            "feature_set",
            "eval_name",
            "feature_count",
            "mean_gain_vs_baseline",
            "mean_gain_vs_static",
            "positive_vs_static_rate",
            "mean_regret_vs_oracle",
            "regret_reduction_vs_static",
            "action_hit_rate",
            "config_hit_rate",
            "fallback_rate",
            "top_features",
        ]
    ]
    out.to_csv(os.path.join(args.output_dir, "phase4_model_aware_router_feature_ablation.csv"), index=False)
    pd.concat(selected_rows, ignore_index=True).to_csv(
        os.path.join(args.output_dir, "phase4_model_aware_router_feature_ablation_selected_features.csv"),
        index=False,
    )
    lines = [
        "# Model-Aware Router Feature Ablation",
        "",
        f"- Cells: {len(cells)}",
        f"- Selection mode: {args.selection_mode}",
        f"- Top-k per set: {args.top_k}",
        "",
        markdown_table(out),
    ]
    with open(os.path.join(args.output_dir, "phase4_model_aware_router_feature_ablation.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print({"output_dir": args.output_dir, "rows": len(out)})


def main():
    parser = argparse.ArgumentParser(description="Feature-set ablation for model-aware activation router")
    parser.add_argument("--factors_dir", required=True)
    parser.add_argument("--output_dir", default="phase4_remote_results/analysis_current/factors_router/model_aware")
    parser.add_argument("--lee_root", default="lee_ocil")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--selection_mode", choices=["train_score", "nested"], default="train_score")
    parser.add_argument("--modes", nargs="+", default=["leave_cell", "leave_dataset"])
    parser.add_argument("--granularity", nargs="+", default=["config", "action_oracle"])
    parser.add_argument("--informer", default="phase4_remote_results/analysis_current/phase4_protocol_aggregate.csv")
    parser.add_argument("--patchtst", default="patchtst_remote_results/analysis_patchtst_transfer_actfix/patchtst_transfer_aggregate.csv")
    parser.add_argument("--itransformer_stage1", default="itransformer_remote_results/analysis_itransformer_stage1_actfix3/itransformer_protocol_aggregate.csv")
    parser.add_argument("--itransformer_placeprobe", default="itransformer_remote_results/analysis_itransformer_placeprobe_20260522/itransformer_protocol_aggregate.csv")
    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()
