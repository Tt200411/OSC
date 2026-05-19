#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from phase4_common import CONFIG_NAMES, expected_matrix, read_all_summaries, reusable_runs, run_key
from phase4_report_utils import markdown_table


def paired_tests(runs):
    rows = []
    base_sig = "act=gelu|enc=gelu|dec=gelu|out=linear|a=0|lee=1"
    base_cols = ["dataset", "pred_len", "seq_len", "label_len", "e_layers", "d_layers", "factor", "seed"]
    lookup = {
        tuple(row[col] for col in base_cols) + (row["activation_signature"],): row
        for row in runs.to_dict("records")
    }
    for row in runs.to_dict("records"):
        if row["activation_signature"] == base_sig:
            continue
        key = tuple(row[col] for col in base_cols)
        baseline = lookup.get(key + (base_sig,))
        if not baseline:
            continue
        rows.append(
            {
                "dataset": row["dataset"],
                "pred_len": row["pred_len"],
                "config_name": row["config_name"],
                "activation_signature": row["activation_signature"],
                "seed": row["seed"],
                "mse": row["mse"],
                "gelu_mse": baseline["mse"],
                "relative_mse_improvement_vs_gelu": (baseline["mse"] - row["mse"]) / baseline["mse"],
                "mae": row["mae"],
                "gelu_mae": baseline["mae"],
            }
        )
    return pd.DataFrame(rows)


def aggregate(runs):
    group_cols = [
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
        "activation",
        "encoder_activation",
        "decoder_activation",
        "output_activation",
        "activation_signature",
    ]
    agg = (
        runs.groupby(group_cols, dropna=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            seed_count=("seed", "nunique"),
            run_count=("setting", "count"),
            has_arrays_count=("has_arrays", "sum"),
        )
        .reset_index()
    )
    paired = paired_tests(runs)
    if not paired.empty:
        paired_agg = (
            paired.groupby(["dataset", "pred_len", "config_name", "activation_signature"], dropna=False)
            .agg(
                improvement_vs_gelu_mean=("relative_mse_improvement_vs_gelu", "mean"),
                improvement_vs_gelu_min=("relative_mse_improvement_vs_gelu", "min"),
                win_rate_vs_gelu=("relative_mse_improvement_vs_gelu", lambda s: float((s > 0).mean())),
                paired_seed_count=("seed", "nunique"),
            )
            .reset_index()
        )
        agg = agg.merge(
            paired_agg,
            on=["dataset", "pred_len", "config_name", "activation_signature"],
            how="left",
        )
    return agg.sort_values(["dataset", "pred_len", "mse_mean"])


def cell_completeness(expected, runs):
    expected = expected.copy()
    runs = runs.copy()
    expected["_run_key"] = expected.apply(run_key, axis=1)
    runs["_run_key"] = runs.apply(run_key, axis=1)
    observed = set(runs["_run_key"])
    expected["complete"] = expected["_run_key"].isin(observed)
    group_cols = ["dataset", "pred_len", "config_name", "activation_signature"]
    return (
        expected.groupby(group_cols, dropna=False)
        .agg(required_seed_runs=("seed", "count"), completed_seed_runs=("complete", "sum"))
        .reset_index()
        .assign(complete_3seeds=lambda df: df["completed_seed_runs"] == df["required_seed_runs"])
    )


def write_claim_summary(path, agg, completeness, paired):
    complete = completeness[completeness["complete_3seeds"]]
    incomplete = completeness[~completeness["complete_3seeds"]]
    complete_agg = agg.merge(
        complete[["dataset", "pred_len", "config_name", "activation_signature"]],
        on=["dataset", "pred_len", "config_name", "activation_signature"],
        how="inner",
    )
    best = (
        complete_agg.sort_values(["dataset", "pred_len", "mse_mean"])
        .groupby(["dataset", "pred_len"], dropna=False)
        .head(1)
    )
    wins = best[best["config_name"] != "gelu_all"].copy()
    lines = [
        "# Phase-4 Claim Summary",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Evidence Status",
        "",
        f"- Complete dataset-horizon-config cells: {len(complete)}",
        f"- Incomplete dataset-horizon-config cells: {len(incomplete)}",
        f"- Dataset-horizon cells with a non-GELU best activation: {wins[['dataset', 'pred_len']].drop_duplicates().shape[0] if not wins.empty else 0}",
        "",
        "## Supported",
        "",
        "Same-protocol activation choice is dataset- and horizon-sensitive where cells have all three seeds. The claim is limited to matched Informer settings and same-environment GELU comparisons.",
        "",
        "## Not Supported",
        "",
        "A universal oscillatory or bounded-activation win is not supported unless the full matrix shows consistent three-seed gains. Small-batch Lee diagnostics remain excluded from main evidence.",
        "",
        "## Dataset-Specific Signals",
        "",
    ]
    if best.empty:
        lines.append("No complete cells available yet.")
    else:
        display = best[
            [
                "dataset",
                "pred_len",
                "config_name",
                "mse_mean",
                "mse_std",
                "seed_count",
                "improvement_vs_gelu_mean",
                "win_rate_vs_gelu",
            ]
        ].copy()
        lines.append(markdown_table(display))
    lines.extend(
        [
            "",
            "## Still Needs Verification",
            "",
            "- Any incomplete cell must stay out of the main table until all three seeds exist or are explicitly marked invalid.",
            "- Same-split factor oracles are diagnostic upper bounds; leave-one-seed-out oracles are the only weak generalization evidence.",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate Phase-4 full-grid results")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--summary_glob", action="append", default=[])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    expected = expected_matrix()
    summaries = read_all_summaries(args.summary_glob or None)
    reusable, missing = reusable_runs(expected, summaries)
    completeness = cell_completeness(expected, reusable)
    paired = paired_tests(reusable)
    agg = aggregate(reusable)

    reusable.to_csv(os.path.join(args.output_dir, "phase4_protocol_raw_dedup.csv"), index=False)
    agg.to_csv(os.path.join(args.output_dir, "phase4_protocol_aggregate.csv"), index=False)
    completeness.to_csv(os.path.join(args.output_dir, "phase4_completion_matrix.csv"), index=False)
    paired.to_csv(os.path.join(args.output_dir, "phase4_paired_vs_gelu.csv"), index=False)
    missing.to_csv(os.path.join(args.output_dir, "phase4_missing_after_aggregate.csv"), index=False)
    write_claim_summary(os.path.join(args.output_dir, "phase4_claim_summary.md"), agg, completeness, paired)
    print(
        json.dumps(
            {
                "reusable_seed_runs": len(reusable),
                "missing_seed_runs": len(missing),
                "aggregate_rows": len(agg),
                "complete_cells": int(completeness["complete_3seeds"].sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
