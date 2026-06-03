#!/usr/bin/env python3
import argparse
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from phase4_report_utils import markdown_table


CELL_COLS = ["dataset", "pred_len", "seq_len", "label_len", "e_layers", "d_layers", "factor"]
STATIC_CONFIGS = ["tanh_all", "softsign_all", "tanh_sin001_all"]


def normalize(frame):
    out = frame.copy()
    for column in [c for c in CELL_COLS if c != "dataset"] + ["seed", "mse", "mae"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def mean_mse(frame, cell, config, seeds):
    rows = frame[
        (frame["config_name"].astype(str) == str(config))
        & (frame["seed"].astype(int).isin(set(int(s) for s in seeds)))
    ]
    for col in CELL_COLS:
        if col == "dataset":
            rows = rows[rows[col].astype(str) == str(cell[col])]
        else:
            rows = rows[rows[col].astype(float) == float(cell[col])]
    if rows.empty:
        return np.nan
    return float(rows["mse"].mean())


def best_config(frame, cell, seeds, configs=None):
    rows = frame[frame["seed"].astype(int).isin(set(int(s) for s in seeds))].copy()
    for col in CELL_COLS:
        if col == "dataset":
            rows = rows[rows[col].astype(str) == str(cell[col])]
        else:
            rows = rows[rows[col].astype(float) == float(cell[col])]
    if configs is not None:
        rows = rows[rows["config_name"].astype(str).isin(set(configs))]
    if rows.empty:
        return "", np.nan
    agg = rows.groupby("config_name", dropna=False).agg(mse=("mse", "mean")).reset_index()
    best = agg.sort_values(["mse", "config_name"]).iloc[0]
    return str(best["config_name"]), float(best["mse"])


def evaluate(raw, train_seed, heldout_seeds, train_static_config):
    rows = []
    cells = raw[CELL_COLS].drop_duplicates().sort_values(CELL_COLS)
    for _, cell in cells.iterrows():
        selected_config, selected_train_mse = best_config(raw, cell, [train_seed])
        heldout_selected_mse = mean_mse(raw, cell, selected_config, heldout_seeds)
        heldout_gelu_mse = mean_mse(raw, cell, "gelu_all", heldout_seeds)
        heldout_train_static_mse = mean_mse(raw, cell, train_static_config, heldout_seeds)
        heldout_static_config, heldout_static_mse = best_config(raw, cell, heldout_seeds, STATIC_CONFIGS)
        heldout_oracle_config, heldout_oracle_mse = best_config(raw, cell, heldout_seeds)
        train_static_selected_config, train_static_selected_mse = best_config(raw, cell, [train_seed], STATIC_CONFIGS)
        rows.append(
            {
                **{col: cell[col] for col in CELL_COLS},
                "train_seed": int(train_seed),
                "heldout_seeds": ",".join(str(int(s)) for s in heldout_seeds),
                "selected_config": selected_config,
                "selected_train_mse": selected_train_mse,
                "heldout_selected_mse": heldout_selected_mse,
                "heldout_gelu_mse": heldout_gelu_mse,
                "train_static_config": train_static_config,
                "heldout_train_static_mse": heldout_train_static_mse,
                "train_seed_best_static_config": train_static_selected_config,
                "train_seed_best_static_mse": train_static_selected_mse,
                "heldout_best_static_config": heldout_static_config,
                "heldout_best_static_mse": heldout_static_mse,
                "heldout_oracle_config": heldout_oracle_config,
                "heldout_oracle_mse": heldout_oracle_mse,
                "gain_vs_gelu": (heldout_gelu_mse - heldout_selected_mse) / heldout_gelu_mse
                if heldout_gelu_mse > 0
                else np.nan,
                "gain_vs_train_static": (heldout_train_static_mse - heldout_selected_mse) / heldout_train_static_mse
                if heldout_train_static_mse > 0
                else np.nan,
                "gain_vs_heldout_best_static": (heldout_static_mse - heldout_selected_mse) / heldout_static_mse
                if heldout_static_mse > 0
                else np.nan,
                "regret_vs_heldout_oracle": (heldout_selected_mse - heldout_oracle_mse) / heldout_oracle_mse
                if heldout_oracle_mse > 0
                else np.nan,
                "config_stable": selected_config == heldout_oracle_config,
                "beats_gelu": heldout_selected_mse < heldout_gelu_mse,
                "beats_train_static": heldout_selected_mse < heldout_train_static_mse,
                "beats_heldout_best_static": heldout_selected_mse < heldout_static_mse,
            }
        )
    return pd.DataFrame(rows)


def summarize(cells):
    if cells.empty:
        return pd.DataFrame()
    rows = []
    for dataset, group in cells.groupby("dataset", dropna=False):
        rows.append(summary_row(str(dataset), group))
    rows.append(summary_row("ALL", cells))
    return pd.DataFrame(rows)


def summary_row(name, group):
    return {
        "dataset": name,
        "cell_count": int(len(group)),
        "mean_gain_vs_gelu": float(group["gain_vs_gelu"].mean()),
        "median_gain_vs_gelu": float(group["gain_vs_gelu"].median()),
        "mean_gain_vs_train_static": float(group["gain_vs_train_static"].mean()),
        "mean_gain_vs_heldout_best_static": float(group["gain_vs_heldout_best_static"].mean()),
        "positive_vs_gelu_rate": float(group["beats_gelu"].mean()),
        "positive_vs_train_static_rate": float(group["beats_train_static"].mean()),
        "positive_vs_heldout_best_static_rate": float(group["beats_heldout_best_static"].mean()),
        "mean_regret_vs_heldout_oracle": float(group["regret_vs_heldout_oracle"].mean()),
        "config_stability_rate": float(group["config_stable"].mean()),
        "top_selected_configs": ";".join(group["selected_config"].value_counts().head(5).index.astype(str)),
        "top_heldout_oracle_configs": ";".join(group["heldout_oracle_config"].value_counts().head(5).index.astype(str)),
    }


def write_report(path, summary, cells):
    lines = [
        "# Phase-5 Oracle-Supplement Selection",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This report uses the seed-2024 new-dataset oracle matrix as the calibration signal and evaluates the selected config on held-out seeds.",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Cells",
        "",
        markdown_table(cells),
        "",
        "## Boundary",
        "",
        "- This is not zero-shot; it is the oracle-supplement/calibrated route.",
        "- A positive result here supports a low-cost calibrated selector story, not a no-oracle zero-shot selector story.",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate seed-2024 oracle-selected configs on held-out seeds")
    parser.add_argument("--raw", required=True, help="phase5_oracle_raw_dedup.csv with all seeds")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--train_seed", type=int, default=2024)
    parser.add_argument("--heldout_seeds", default="2025,2026")
    parser.add_argument("--train_static_config", default="tanh_sin001_all")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    raw = normalize(pd.read_csv(args.raw))
    heldout = [int(item) for item in args.heldout_seeds.split(",") if item.strip()]
    cells = evaluate(raw, args.train_seed, heldout, args.train_static_config)
    summary = summarize(cells)
    cells.to_csv(os.path.join(args.output_dir, "phase5_oracle_supplement_selection_cells.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, "phase5_oracle_supplement_selection_summary.csv"), index=False)
    write_report(os.path.join(args.output_dir, "phase5_oracle_supplement_selection_report.md"), summary, cells)
    print({"cells": int(len(cells)), "summary_rows": int(len(summary)), "output_dir": args.output_dir})


if __name__ == "__main__":
    main()
