#!/usr/bin/env python3
import argparse
import os

import numpy as np
import pandas as pd

from phase4_report_utils import markdown_table


def best_by_cell(df, model_name, baseline_config):
    idx = df.groupby(["dataset", "pred_len"], dropna=False)["mse_mean"].idxmin()
    best = df.loc[idx, ["dataset", "pred_len", "config_name", "mse_mean"]].rename(
        columns={"config_name": "best_config", "mse_mean": "best_mse"}
    )
    base = df[df["config_name"] == baseline_config][["dataset", "pred_len", "mse_mean"]].rename(
        columns={"mse_mean": "baseline_mse"}
    )
    out = best.merge(base, on=["dataset", "pred_len"], how="left")
    out["model"] = model_name
    out["baseline_config"] = baseline_config
    out["best_gain_vs_baseline"] = (out["baseline_mse"] - out["best_mse"]) / out["baseline_mse"]
    out["non_baseline_best"] = out["best_config"] != baseline_config
    return out


def static_rank(df, model_name):
    tmp = df.copy()
    tmp["rank_in_cell"] = tmp.groupby(["dataset", "pred_len"], dropna=False)["mse_mean"].rank(method="min")
    rank = (
        tmp.groupby("config_name", dropna=False)
        .agg(
            model=("config_name", lambda _: model_name),
            mean_rank=("rank_in_cell", "mean"),
            best_cells=("rank_in_cell", lambda s: int((s == 1).sum())),
            mean_mse=("mse_mean", "mean"),
            cell_count=("rank_in_cell", "size"),
        )
        .reset_index()
        .sort_values(["mean_rank", "config_name"])
    )
    return rank


def paired_vs_baseline(df, model_name, baseline_config):
    base = df[df["config_name"] == baseline_config][["dataset", "pred_len", "mse_mean"]].rename(
        columns={"mse_mean": "baseline_mse"}
    )
    paired = df.merge(base, on=["dataset", "pred_len"], how="inner")
    paired["model"] = model_name
    paired["baseline_config"] = baseline_config
    paired["gain_vs_baseline"] = (paired["baseline_mse"] - paired["mse_mean"]) / paired["baseline_mse"]
    paired["win_vs_baseline"] = paired["mse_mean"] < paired["baseline_mse"]
    return paired


def summarize_model(best, paired, static):
    rows = []
    for model, group in best.groupby("model", dropna=False):
        paired_model = paired[paired["model"] == model]
        rows.append(
            {
                "model": model,
                "cell_count": int(group[["dataset", "pred_len"]].drop_duplicates().shape[0]),
                "non_baseline_best_cells": int(group["non_baseline_best"].sum()),
                "mean_best_gain_vs_baseline": float(group["best_gain_vs_baseline"].mean()),
                "best_static_config": static[static["model"] == model].iloc[0]["config_name"],
                "best_static_mean_rank": float(static[static["model"] == model].iloc[0]["mean_rank"]),
                "nonbaseline_win_rate": float(
                    paired_model[paired_model["config_name"] != paired_model["baseline_config"]]["win_vs_baseline"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def normalize_itransformer_placeprobe(path):
    df = pd.read_csv(path)
    if "mse_mean" not in df.columns:
        df = (
            df.groupby(["dataset", "pred_len", "config_name", "placement"], dropna=False)
            .agg(mse_mean=("mse", "mean"), seed_count=("seed", "nunique"))
            .reset_index()
        )
    return df


def write_summary(path, model_summary, static, best, placement, factor_join):
    lines = [
        "# Cross-Architecture Router Context",
        "",
        "## Model-Level Summary",
        "",
        markdown_table(model_summary),
        "",
        "## Static Ranking",
        "",
        markdown_table(static.groupby("model", group_keys=False).head(5)),
        "",
        "## Best Cell Configs",
        "",
        markdown_table(best.sort_values(["model", "dataset", "pred_len"])),
    ]
    if placement is not None and not placement.empty:
        lines.extend(["", "## Placement Signals", "", markdown_table(placement)])
    if factor_join is not None and not factor_join.empty:
        lines.extend(
            [
                "",
                "## Factor-Joined Cross-Architecture Cells",
                "",
                markdown_table(factor_join.head(30)),
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Informer has a very strong static tanh-family policy; this makes an Informer-only factor router hard to beat.",
            "- PatchTST is a negative transfer stress test: GELU-linear is best in most cells and output-tanh is consistently harmful in the tested placement matrix.",
            "- iTransformer shows a different pattern: FFN-only tanh-family does not transfer broadly, but small placement/regime probes beat GELU-linear in all three tested cells.",
            "- Therefore the next router target should not be 'beat Informer tanh_sin001 on Informer'. It should be architecture-aware or cross-architecture: decide when tanh-family transfers, when GELU should remain, and when output placement is worth testing.",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Summarize cross-architecture evidence for factor-router framing")
    parser.add_argument("--informer", default="phase4_remote_results/analysis_current/phase4_protocol_aggregate.csv")
    parser.add_argument("--patchtst", default="patchtst_remote_results/analysis_patchtst_transfer_actfix/patchtst_transfer_aggregate.csv")
    parser.add_argument("--itransformer_stage1", default="itransformer_remote_results/analysis_itransformer_stage1_actfix3/itransformer_protocol_aggregate.csv")
    parser.add_argument("--itransformer_placeprobe", default="itransformer_remote_results/analysis_itransformer_placeprobe_20260522/itransformer_protocol_aggregate.csv")
    parser.add_argument("--router_factors", default="phase4_remote_results/analysis_current/factors_router/phase4_router_factor_table.csv")
    parser.add_argument("--output_dir", default="phase4_remote_results/analysis_current/factors_router")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    informer = pd.read_csv(args.informer)
    patchtst = pd.read_csv(args.patchtst)
    itrans_stage1 = pd.read_csv(args.itransformer_stage1)
    itrans_place = normalize_itransformer_placeprobe(args.itransformer_placeprobe)

    best = pd.concat(
        [
            best_by_cell(informer, "Informer", "gelu_all"),
            best_by_cell(patchtst, "PatchTST", "gelu_ffn_linear"),
            best_by_cell(itrans_stage1, "iTransformer-stage1", "gelu_ffn_linear"),
            best_by_cell(itrans_place, "iTransformer-placeprobe", "gelu_ffn_linear"),
        ],
        ignore_index=True,
    )
    static = pd.concat(
        [
            static_rank(informer, "Informer"),
            static_rank(patchtst, "PatchTST"),
            static_rank(itrans_stage1, "iTransformer-stage1"),
            static_rank(itrans_place, "iTransformer-placeprobe"),
        ],
        ignore_index=True,
    )
    paired = pd.concat(
        [
            paired_vs_baseline(informer, "Informer", "gelu_all"),
            paired_vs_baseline(patchtst, "PatchTST", "gelu_ffn_linear"),
            paired_vs_baseline(itrans_stage1, "iTransformer-stage1", "gelu_ffn_linear"),
            paired_vs_baseline(itrans_place, "iTransformer-placeprobe", "gelu_ffn_linear"),
        ],
        ignore_index=True,
    )
    model_summary = summarize_model(best, paired, static)

    placement_rows = []
    if {"placement", "mse_mean"}.issubset(patchtst.columns):
        for model, df, baseline in [
            ("PatchTST", patchtst, "ffn_linear"),
            ("iTransformer-placeprobe", itrans_place, "ffn_linear"),
        ]:
            if "placement" not in df.columns:
                continue
            tmp = df.copy()
            tmp["is_outtanh"] = tmp["placement"].astype(str).str.contains("outtanh")
            placement_rows.append(
                {
                    "model": model,
                    "outtanh_best_cells": int(
                        best[best["model"] == model]["best_config"].astype(str).str.contains("outtanh").sum()
                    ),
                    "outtanh_rows": int(tmp["is_outtanh"].sum()),
                    "linear_rows": int((~tmp["is_outtanh"]).sum()),
                }
            )
    placement = pd.DataFrame(placement_rows)

    factor_join = pd.DataFrame()
    if os.path.exists(args.router_factors):
        factors = pd.read_csv(args.router_factors)
        keep = [
            "dataset",
            "pred_len",
            "input_cv",
            "input_spectral_entropy",
            "input_dominant_frequency_energy_ratio",
            "router_multivar_channel_heterogeneity_cv",
            "router_timefreq_burstiness_index",
            "router_freq_seasonal_peak_strength",
        ]
        keep = [col for col in keep if col in factors.columns]
        factor_join = best.merge(factors[keep], on=["dataset", "pred_len"], how="left")

    model_summary.to_csv(os.path.join(args.output_dir, "phase4_cross_arch_model_summary.csv"), index=False)
    static.to_csv(os.path.join(args.output_dir, "phase4_cross_arch_static_ranking.csv"), index=False)
    best.to_csv(os.path.join(args.output_dir, "phase4_cross_arch_best_cells.csv"), index=False)
    paired.to_csv(os.path.join(args.output_dir, "phase4_cross_arch_paired_vs_baseline.csv"), index=False)
    placement.to_csv(os.path.join(args.output_dir, "phase4_cross_arch_placement_summary.csv"), index=False)
    factor_join.to_csv(os.path.join(args.output_dir, "phase4_cross_arch_factor_join.csv"), index=False)
    write_summary(
        os.path.join(args.output_dir, "phase4_cross_arch_router_context.md"),
        model_summary,
        static,
        best,
        placement,
        factor_join,
    )
    print({"output_dir": args.output_dir, "models": int(model_summary.shape[0]), "best_rows": int(best.shape[0])})


if __name__ == "__main__":
    main()
