#!/usr/bin/env python3
import argparse
import os
import subprocess
import shutil

import pandas as pd

from phase4_report_utils import markdown_table


def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def select_array_ready_runs(protocol_raw, min_seed_count):
    df = pd.read_csv(protocol_raw)
    df = df[df["has_arrays"].astype(bool)].copy()
    if df.empty:
        return df
    group_cols = ["dataset", "pred_len", "config_name", "activation_signature"]
    seed_counts = df.groupby(group_cols, dropna=False)["seed"].nunique().reset_index(name="array_seed_count")
    eligible = seed_counts[seed_counts["array_seed_count"] >= min_seed_count]
    return df.merge(eligible[group_cols], on=group_cols, how="inner")


def build_result_links(eligible, output_dir):
    link_dir = os.path.join(output_dir, "phase4_factor_results_links")
    os.makedirs(link_dir, exist_ok=True)
    linked_rows = []
    used_names = set()
    for row in eligible.to_dict("records"):
        source_dir = os.path.dirname(row.get("config_path", ""))
        if not source_dir or not os.path.isdir(source_dir):
            continue
        setting = str(row["setting"])
        link_name = setting
        if link_name in used_names:
            suffix = len(used_names)
            link_name = f"{setting}__dup{suffix}"
        used_names.add(link_name)
        link_path = os.path.join(link_dir, link_name)
        if os.path.lexists(link_path):
            if os.path.islink(link_path) and os.path.realpath(link_path) == os.path.realpath(source_dir):
                pass
            else:
                continue
        else:
            os.symlink(os.path.abspath(source_dir), link_path)
        row["setting"] = link_name
        linked_rows.append(row)
    return pd.DataFrame(linked_rows), link_dir


def main():
    parser = argparse.ArgumentParser(description="Run Phase-4 factor-bin and oracle analyses")
    parser.add_argument("--protocol_raw", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--factors_dir", required=True)
    parser.add_argument("--lee_root", default="lee_ocil")
    parser.add_argument("--min_seed_count", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sample_path = os.path.join(args.output_dir, "phase4_factor_sample_results.csv")
    bin_path = os.path.join(args.output_dir, "phase4_factor_bin_summary.csv")
    contrast_path = os.path.join(args.output_dir, "phase4_factor_contrast_summary.csv")
    eligible = select_array_ready_runs(args.protocol_raw, args.min_seed_count)
    eligible, linked_results_dir = build_result_links(eligible, args.output_dir)
    eligible_summary = os.path.join(args.output_dir, "phase4_array_ready_summary.csv")
    eligible.to_csv(eligible_summary, index=False)
    if eligible.empty:
        for name in [
            "phase4_factor_sample_results.csv",
            "phase4_factor_bin_summary.csv",
            "phase4_factor_contrast_summary.csv",
            "phase4_static_activation_ranking.csv",
            "phase4_factor_oracle_summary.csv",
            "phase4_factor_oracle_loso_summary.csv",
            "phase4_factor_bin_best_activation.csv",
        ]:
            pd.DataFrame().to_csv(os.path.join(args.output_dir, name), index=False)
        with open(os.path.join(args.output_dir, "phase4_factor_claim_summary.md"), "w", encoding="utf-8") as handle:
            handle.write("# Phase-4 Factor Claim Summary\n\nNo three-seed array-complete cells are available yet.\n")
        print("No array-complete cells available")
        return

    factor_work = os.path.join(args.output_dir, "_factor_work")
    oracle_work = os.path.join(args.output_dir, "_oracle_work")
    os.makedirs(factor_work, exist_ok=True)
    os.makedirs(oracle_work, exist_ok=True)
    should_run_factor = args.force or not (
        os.path.exists(sample_path) and os.path.exists(bin_path) and os.path.exists(contrast_path)
    )
    if should_run_factor:
        run(
            [
                "python",
                "lee_ocil/scripts/analyze_factor_effects.py",
                eligible_summary,
                "--results_dir",
                linked_results_dir,
                "--factors_dir",
                args.factors_dir,
                "--lee_root",
                args.lee_root,
                "--output_dir",
                factor_work,
            ]
        )
        sample_src = os.path.join(factor_work, "phase1_factor_sample_results.csv")
        bin_src = os.path.join(factor_work, "phase1_factor_bin_summary.csv")
        contrast_src = os.path.join(factor_work, "phase1_factor_contrast_summary.csv")
        shutil.copy2(sample_src, sample_path)
        shutil.copy2(bin_src, bin_path)
        shutil.copy2(contrast_src, contrast_path)

    run(
        [
            "python",
            "lee_ocil/scripts/analyze_regime_activation_oracle.py",
            sample_path,
            "--output_dir",
            oracle_work,
        ]
    )
    rename = {
        "static_activation_ranking.csv": "phase4_static_activation_ranking.csv",
        "factor_oracle_summary.csv": "phase4_factor_oracle_summary.csv",
        "factor_oracle_loso_summary.csv": "phase4_factor_oracle_loso_summary.csv",
        "factor_bin_best_activation.csv": "phase4_factor_bin_best_activation.csv",
    }
    for src, dst in rename.items():
        shutil.copy2(os.path.join(oracle_work, src), os.path.join(args.output_dir, dst))

    oracle = pd.read_csv(os.path.join(args.output_dir, "phase4_factor_oracle_summary.csv"))
    loso = pd.read_csv(os.path.join(args.output_dir, "phase4_factor_oracle_loso_summary.csv"))
    lines = [
        "# Phase-4 Factor Claim Summary",
        "",
        f"- Array-ready seed-runs: {len(eligible)}",
        f"- Same-split oracle rows: {len(oracle)}",
        f"- LOSO oracle rows: {len(loso)}",
        "",
        "Same-split oracle results are diagnostic upper bounds. Treat only LOSO gains as weak generalization evidence.",
    ]
    if not loso.empty and "loso_gain_vs_best_static" in loso.columns:
        top = loso.sort_values("loso_gain_vs_best_static", ascending=False).head(10)
        lines.extend(["", "## Strongest LOSO Signals", "", markdown_table(top)])
    with open(os.path.join(args.output_dir, "phase4_factor_claim_summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
