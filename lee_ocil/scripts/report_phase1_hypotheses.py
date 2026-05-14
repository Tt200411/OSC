#!/usr/bin/env python3
"""Create compact hypothesis evidence tables from Phase 1 summaries."""

from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable

import pandas as pd


KEY_COLUMNS = [
    "dataset",
    "pred_len",
    "features",
    "activation",
    "activation_family",
    "amplitude",
    "lee_type",
]


def expand_paths(paths: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for path in paths:
        matches = glob.glob(path)
        expanded.extend(matches if matches else [path])
    return expanded


def read_csvs(paths: Iterable[str]) -> pd.DataFrame:
    frames = []
    for path in expand_paths(paths):
        if os.path.isdir(path):
            path = os.path.join(path, "phase1_activation_summary.csv")
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path)
        frame["source_file"] = path
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_factor_csvs(paths: Iterable[str]) -> pd.DataFrame:
    frames = []
    for path in expand_paths(paths):
        if os.path.isdir(path):
            path = os.path.join(path, "phase1_factor_contrast_summary.csv")
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path)
        frame["factor_source_file"] = path
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce")


def classify_global(row: pd.Series) -> str:
    seed_count = row.get("seed_count", pd.NA)
    change = row.get("relative_mse_change_mean", pd.NA)
    win_rate = row.get("win_rate", pd.NA)
    if pd.isna(change) or pd.isna(win_rate):
        return "baseline_or_unpaired"
    if not pd.isna(seed_count) and seed_count < 3:
        prefix = "single_seed_" if seed_count == 1 else "low_seed_"
    else:
        prefix = ""
    if change >= 0.05 and win_rate >= 0.8:
        return prefix + "strong_effective"
    if change >= 0.02 and win_rate >= 0.6:
        return prefix + "effective"
    if change <= -0.02:
        return prefix + "negative"
    if change < 0 or win_rate < 0.5:
        return prefix + "not_supported"
    return prefix + "weak_signal"


def classify_factor(row: pd.Series) -> str:
    high_effect = row.get("target_relative_change_high", pd.NA)
    high_win = row.get("target_win_rate_high", pd.NA)
    gain = row.get("target_relative_change_high_minus_low", pd.NA)
    win_gain = row.get("target_win_rate_high_minus_low", pd.NA)
    spearman = row.get("relative_target_mse_change_spearman", pd.NA)
    if pd.isna(gain) and pd.isna(win_gain) and pd.isna(spearman):
        return "insufficient_bins"
    supports_gradient = (
        (not pd.isna(gain) and gain >= 0.03)
        or (not pd.isna(win_gain) and win_gain >= 0.10)
        or (not pd.isna(spearman) and spearman >= 0.10)
    )
    opposes_gradient = (
        (not pd.isna(gain) and gain <= -0.03)
        or (not pd.isna(win_gain) and win_gain <= -0.10)
        or (not pd.isna(spearman) and spearman <= -0.10)
    )
    high_bin_effective = (
        not pd.isna(high_effect)
        and high_effect >= 0.02
        and (pd.isna(high_win) or high_win >= 0.55)
    )
    if supports_gradient and high_bin_effective:
        return "supports_high_factor_benefit"
    if supports_gradient:
        return "supports_factor_gradient_only"
    if opposes_gradient:
        return "opposes_high_factor_benefit"
    return "flat_or_mixed"


def build_report(summary: pd.DataFrame, factors: pd.DataFrame, focus_factors: set[str]) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    summary = summary.copy()
    summary = summary[summary["activation"] != summary.get("baseline_activation", "")]
    summary["relative_mse_change_mean"] = numeric(summary, "relative_mse_change_mean")
    summary["win_rate"] = numeric(summary, "win_rate")
    summary["seed_count"] = numeric(summary, "seed_count")
    summary["global_label"] = summary.apply(classify_global, axis=1)

    rows = []
    if factors.empty:
        for _, row in summary.iterrows():
            output = row.to_dict()
            output["factor"] = pd.NA
            output["factor_label"] = "not_available"
            rows.append(output)
        return pd.DataFrame(rows)

    factors = factors.copy()
    factors = factors[factors["factor"].isin(focus_factors)]
    key_cols = [col for col in KEY_COLUMNS if col in summary.columns and col in factors.columns]
    factor_cols = [
        "factor",
        "sample_count",
        "seed_count",
        "start_bin",
        "middle_bin",
        "end_bin",
        "relative_target_mse_change_mean_start",
        "relative_target_mse_change_mean_middle",
        "relative_target_mse_change_mean_end",
        "relative_target_mse_change_mean_end_minus_start",
        "target_win_rate_start",
        "target_win_rate_middle",
        "target_win_rate_end",
        "target_win_rate_end_minus_start",
        "relative_target_mse_change_spearman",
        "target_win_spearman",
        "factor_conditioned",
        "factor_source_file",
    ]
    factor_cols = [col for col in factor_cols if col in factors.columns]
    merged = summary.merge(factors[key_cols + factor_cols], on=key_cols, how="left")
    rename = {
        "relative_target_mse_change_mean_start": "target_relative_change_low",
        "relative_target_mse_change_mean_middle": "target_relative_change_mid",
        "relative_target_mse_change_mean_end": "target_relative_change_high",
        "relative_target_mse_change_mean_end_minus_start": "target_relative_change_high_minus_low",
        "target_win_rate_start": "target_win_rate_low",
        "target_win_rate_middle": "target_win_rate_mid",
        "target_win_rate_end": "target_win_rate_high",
        "target_win_rate_end_minus_start": "target_win_rate_high_minus_low",
    }
    merged = merged.rename(columns=rename)
    for column in [
        "target_relative_change_low",
        "target_relative_change_mid",
        "target_relative_change_high",
        "target_relative_change_high_minus_low",
        "target_win_rate_low",
        "target_win_rate_mid",
        "target_win_rate_high",
        "target_win_rate_high_minus_low",
        "relative_target_mse_change_spearman",
        "target_win_spearman",
    ]:
        if column in merged:
            merged[column] = numeric(merged, column)
    merged["factor_label"] = merged.apply(classify_factor, axis=1)
    return merged


def write_markdown(report: pd.DataFrame, output_path: str) -> None:
    lines = ["# Phase 1 Hypothesis Evidence", ""]
    if report.empty:
        lines.append("No evidence rows were generated.")
    else:
        focus = report[
            report["factor_label"].isin(
                ["supports_high_factor_benefit", "supports_factor_gradient_only"]
            )
        ].copy()
        if focus.empty:
            lines.append("No factor-gradient support rows passed the current thresholds.")
        else:
            lines.append("Rows supporting factor-conditioned oscillation:")
            lines.append("")
            for _, row in focus.sort_values(
                ["dataset", "pred_len", "activation", "amplitude", "factor"]
            ).iterrows():
                lines.append(
                    "- {dataset} pl{pred_len} {activation} a={amplitude} lee={lee_type} "
                    "{factor}: high-low target change {gain:.4f}, high-low win {win_gain:.4f}, "
                    "global {global_label}, factor {factor_label}".format(
                        dataset=row.get("dataset"),
                        pred_len=row.get("pred_len"),
                        activation=row.get("activation"),
                        amplitude=row.get("amplitude"),
                        lee_type=row.get("lee_type"),
                        factor=row.get("factor"),
                        gain=row.get("target_relative_change_high_minus_low", float("nan")),
                        win_gain=row.get("target_win_rate_high_minus_low", float("nan")),
                        global_label=row.get("global_label"),
                        factor_label=row.get("factor_label"),
                    )
                )
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation_summary", action="append", required=True)
    parser.add_argument("--factor_contrast", action="append", default=[])
    parser.add_argument("--output_dir", default="analysis_hypotheses")
    parser.add_argument(
        "--focus_factors",
        default="volatility,turbulence_score",
        help="Comma-separated factors to include in the report.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary = read_csvs(args.activation_summary)
    factors = read_factor_csvs(args.factor_contrast)
    focus_factors = {item.strip() for item in args.focus_factors.split(",") if item.strip()}
    report = build_report(summary, factors, focus_factors)
    report_path = os.path.join(args.output_dir, "phase1_hypothesis_evidence.csv")
    markdown_path = os.path.join(args.output_dir, "phase1_hypothesis_evidence.md")
    report.to_csv(report_path, index=False)
    write_markdown(report, markdown_path)
    print(f"Wrote {report_path}: {len(report)} rows")
    print(f"Wrote {markdown_path}")


if __name__ == "__main__":
    main()
