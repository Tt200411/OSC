#!/usr/bin/env python3
import argparse
import os

import numpy as np
import pandas as pd

from analyze_factor_effects import FACTOR_COLUMNS, factor_rows_for_prediction
from phase4_common import data_path_for_dataset, root_path_for_dataset
from phase4_report_utils import markdown_table


BIN_ORDER = {"low": 0, "mid": 1, "high": 2, "down": 0, "flat": 1, "up": 2}


def prediction_sample_count(dataset, seq_len, pred_len, config, lee_root):
    if dataset in {"ETTh1", "ETTh2"}:
        border1 = 12 * 30 * 24 + 4 * 30 * 24 - seq_len
        border2 = 12 * 30 * 24 + 8 * 30 * 24
    elif dataset in {"ETTm1", "ETTm2"}:
        border1 = 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len
        border2 = 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4
    else:
        csv_path = os.path.normpath(os.path.join(lee_root, config["root_path"], config["data_path"]))
        row_count = len(pd.read_csv(csv_path))
        num_test = int(row_count * 0.2)
        border1 = row_count - num_test - seq_len
        border2 = row_count
    return max(0, border2 - border1 - seq_len - pred_len + 1)


def bin_series(series, factor):
    valid = series.dropna()
    if valid.empty:
        return pd.Series(pd.NA, index=series.index, dtype="object")
    labels = ["down", "flat", "up"] if factor == "trend" else ["low", "mid", "high"]
    if valid.nunique() < 3:
        median = valid.median()
        fallback = ["down", "up"] if factor == "trend" else ["low", "high"]
        return pd.Series(np.where(series <= median, fallback[0], fallback[1]), index=series.index)
    return pd.qcut(series.rank(method="first"), q=3, labels=labels).astype("object")


def build_cell_factor_table(agg, factors_dir, lee_root):
    cells = agg[
        ["dataset", "pred_len", "seq_len", "label_len", "e_layers", "d_layers", "factor"]
    ].drop_duplicates()
    rows = []
    for cell in cells.to_dict("records"):
        dataset = cell["dataset"]
        pred_len = int(cell["pred_len"])
        seq_len = int(cell["seq_len"])
        config = {
            "data": dataset,
            "seq_len": seq_len,
            "pred_len": pred_len,
            "root_path": root_path_for_dataset(dataset),
            "data_path": data_path_for_dataset(dataset),
        }
        factors_path = os.path.join(factors_dir, f"{dataset}_factors.csv")
        factors = pd.read_csv(factors_path)
        n_samples = prediction_sample_count(dataset, seq_len, pred_len, config, lee_root)
        factor_rows = factor_rows_for_prediction(config, factors, n_samples, lee_root)
        summary = {**cell, "test_sample_count": n_samples}
        for factor in FACTOR_COLUMNS:
            summary[f"input_{factor}"] = factor_rows[f"input_{factor}"].mean()
            summary[f"target_{factor}"] = factor_rows[f"target_{factor}"].mean()
        rows.append(summary)
    return pd.DataFrame(rows)


def annotate_bins(cell_factors):
    out = cell_factors.copy()
    for factor in FACTOR_COLUMNS:
        out[f"{factor}_bin"] = bin_series(out[f"target_{factor}"], factor)
    return out


def result_rows(agg, cell_factors):
    keys = ["dataset", "pred_len", "seq_len", "label_len", "e_layers", "d_layers", "factor"]
    df = agg.merge(cell_factors, on=keys, how="left")
    df["mse_rank_in_cell"] = df.groupby(["dataset", "pred_len"])["mse_mean"].rank(method="min")
    df["best_in_cell"] = df["mse_rank_in_cell"] == 1
    df["top3_in_cell"] = df["mse_rank_in_cell"] <= 3
    df["improvement_for_summary"] = df["improvement_vs_gelu_mean"]
    df.loc[df["config_name"] == "gelu_all", "improvement_for_summary"] = 0.0
    df["positive_vs_gelu"] = df["improvement_for_summary"] > 0
    return df


def factor_bin_summary(df):
    rows = []
    group_cols = [
        "factor",
        "factor_bin",
        "config_name",
        "activation",
        "encoder_activation",
        "decoder_activation",
        "output_activation",
        "activation_signature",
    ]
    for factor in FACTOR_COLUMNS:
        bin_col = f"{factor}_bin"
        tmp = df[df[bin_col].notna()].copy()
        tmp["factor"] = factor
        tmp["factor_bin"] = tmp[bin_col].astype(str)
        grouped = (
            tmp.groupby(group_cols, dropna=False)
            .agg(
                cell_count=("dataset", "count"),
                dataset_count=("dataset", "nunique"),
                mse_mean=("mse_mean", "mean"),
                mse_median=("mse_mean", "median"),
                mean_rank_in_cell=("mse_rank_in_cell", "mean"),
                best_cell_count=("best_in_cell", "sum"),
                top3_cell_count=("top3_in_cell", "sum"),
                improvement_vs_gelu_mean=("improvement_for_summary", "mean"),
                improvement_vs_gelu_median=("improvement_for_summary", "median"),
                positive_cell_rate=("positive_vs_gelu", "mean"),
                win_rate_vs_gelu_mean=("win_rate_vs_gelu", "mean"),
            )
            .reset_index()
        )
        rows.append(grouped)
    out = pd.concat(rows, ignore_index=True)
    out["bin_order"] = out["factor_bin"].map(BIN_ORDER).fillna(99).astype(int)
    return out.sort_values(["factor", "bin_order", "mean_rank_in_cell", "config_name"]).drop(columns=["bin_order"])


def factor_contrast_summary(bin_summary):
    rows = []
    id_cols = [
        "factor",
        "config_name",
        "activation",
        "encoder_activation",
        "decoder_activation",
        "output_activation",
        "activation_signature",
    ]
    for _, group in bin_summary.groupby(id_cols, dropna=False):
        factor = group["factor"].iloc[0]
        ordered_bins = ["down", "flat", "up"] if factor == "trend" else ["low", "mid", "high"]
        by_bin = {row["factor_bin"]: row for row in group.to_dict("records")}
        start = by_bin.get(ordered_bins[0], {})
        middle = by_bin.get(ordered_bins[1], {})
        end = by_bin.get(ordered_bins[2], {})
        row = {col: group[col].iloc[0] for col in id_cols}
        row.update(
            {
                "start_bin": ordered_bins[0],
                "middle_bin": ordered_bins[1],
                "end_bin": ordered_bins[2],
                "cell_count_start": start.get("cell_count", 0),
                "cell_count_middle": middle.get("cell_count", 0),
                "cell_count_end": end.get("cell_count", 0),
                "improvement_start": start.get("improvement_vs_gelu_mean", np.nan),
                "improvement_middle": middle.get("improvement_vs_gelu_mean", np.nan),
                "improvement_end": end.get("improvement_vs_gelu_mean", np.nan),
                "win_rate_start": start.get("win_rate_vs_gelu_mean", np.nan),
                "win_rate_middle": middle.get("win_rate_vs_gelu_mean", np.nan),
                "win_rate_end": end.get("win_rate_vs_gelu_mean", np.nan),
                "rank_start": start.get("mean_rank_in_cell", np.nan),
                "rank_middle": middle.get("mean_rank_in_cell", np.nan),
                "rank_end": end.get("mean_rank_in_cell", np.nan),
            }
        )
        row["improvement_end_minus_start"] = row["improvement_end"] - row["improvement_start"]
        row["win_rate_end_minus_start"] = row["win_rate_end"] - row["win_rate_start"]
        row["rank_end_minus_start"] = row["rank_end"] - row["rank_start"]
        row["factor_conditioned"] = bool(
            pd.notna(row["improvement_end_minus_start"]) and abs(row["improvement_end_minus_start"]) >= 0.02
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["factor", "config_name"])


def static_activation_ranking(df):
    grouped = (
        df.groupby(
            [
                "config_name",
                "activation",
                "encoder_activation",
                "decoder_activation",
                "output_activation",
                "activation_signature",
            ],
            dropna=False,
        )
        .agg(
            cell_count=("dataset", "count"),
            mean_rank_in_cell=("mse_rank_in_cell", "mean"),
            median_rank_in_cell=("mse_rank_in_cell", "median"),
            best_cell_count=("best_in_cell", "sum"),
            top3_cell_count=("top3_in_cell", "sum"),
            improvement_vs_gelu_mean=("improvement_for_summary", "mean"),
            improvement_vs_gelu_median=("improvement_for_summary", "median"),
            positive_cell_rate=("positive_vs_gelu", "mean"),
            win_rate_vs_gelu_mean=("win_rate_vs_gelu", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(["mean_rank_in_cell", "improvement_vs_gelu_mean"], ascending=[True, False])


def factor_bin_best_activation(bin_summary):
    candidates = bin_summary[bin_summary["config_name"] != "gelu_all"].copy()
    candidates = candidates.sort_values(
        ["factor", "factor_bin", "mean_rank_in_cell", "improvement_vs_gelu_mean"],
        ascending=[True, True, True, False],
    )
    return candidates.groupby(["factor", "factor_bin"], dropna=False).head(1).reset_index(drop=True)


def write_claim_summary(path, bin_summary, contrast, static_rank, cell_factors):
    best = static_rank.head(5)[
        ["config_name", "mean_rank_in_cell", "best_cell_count", "improvement_vs_gelu_mean", "positive_cell_rate"]
    ]
    conditioned = contrast[contrast["factor_conditioned"]].copy()
    lines = [
        "# Phase-4 Factor Claim Summary",
        "",
        f"- Summary-level factor-bin coverage: {cell_factors[['dataset', 'pred_len']].drop_duplicates().shape[0]} dataset-horizon cells.",
        f"- Factor-bin rows: {len(bin_summary)}.",
        f"- Factor-conditioned contrasts with |end-start improvement| >= 0.02: {len(conditioned)}.",
        "",
        "Summary-level factor bins cover the full completed matrix. Sample-level oracle files in this directory remain diagnostic and are restricted to rows with prediction arrays.",
        "",
        "## Static Activation Ranking",
        "",
        markdown_table(best),
    ]
    if not conditioned.empty:
        cols = [
            "factor",
            "config_name",
            "start_bin",
            "end_bin",
            "improvement_start",
            "improvement_end",
            "improvement_end_minus_start",
        ]
        lines.extend(["", "## Strongest Factor-Conditioned Contrasts", ""])
        lines.append(markdown_table(conditioned.reindex(conditioned["improvement_end_minus_start"].abs().sort_values(ascending=False).index).head(10)[cols]))
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Full-matrix summary-level Phase-4 factor-bin analysis")
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--factors_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lee_root", default="lee_ocil")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    agg = pd.read_csv(args.aggregate)
    cell_factors = annotate_bins(build_cell_factor_table(agg, args.factors_dir, args.lee_root))
    rows = result_rows(agg, cell_factors)
    bins = factor_bin_summary(rows)
    contrasts = factor_contrast_summary(bins)
    static_rank = static_activation_ranking(rows)
    best = factor_bin_best_activation(bins)

    cell_factors.to_csv(os.path.join(args.output_dir, "phase4_factor_cell_summary.csv"), index=False)
    bins.to_csv(os.path.join(args.output_dir, "phase4_factor_bin_summary.csv"), index=False)
    contrasts.to_csv(os.path.join(args.output_dir, "phase4_factor_contrast_summary.csv"), index=False)
    static_rank.to_csv(os.path.join(args.output_dir, "phase4_static_activation_ranking.csv"), index=False)
    best.to_csv(os.path.join(args.output_dir, "phase4_factor_bin_best_activation.csv"), index=False)
    write_claim_summary(
        os.path.join(args.output_dir, "phase4_factor_claim_summary.md"),
        bins,
        contrasts,
        static_rank,
        cell_factors,
    )
    print(
        {
            "cell_factor_rows": len(cell_factors),
            "factor_bin_rows": len(bins),
            "factor_contrast_rows": len(contrasts),
            "static_rank_rows": len(static_rank),
            "best_bin_rows": len(best),
        }
    )


if __name__ == "__main__":
    main()
