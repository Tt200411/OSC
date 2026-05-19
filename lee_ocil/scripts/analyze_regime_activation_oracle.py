import argparse
import os

import numpy as np
import pandas as pd


DEFAULT_FACTORS = [
    "volatility",
    "std",
    "range",
    "max_abs_change",
    "trend",
    "realized_volatility",
    "volatility_shock",
    "trend_consistency",
    "reversal_rate",
    "jump_intensity",
    "tail_ratio",
    "mean_reversion_strength",
    "max_drawdown",
    "max_drawup",
]

SAMPLE_KEY_CANDIDATES = [
    "dataset",
    "pred_len",
    "features",
    "target",
    "seq_len",
    "label_len",
    "e_layers",
    "d_layers",
    "factor",
    "batch_size",
    "train_epochs",
    "seed",
    "sample_index",
]


def split_values(values):
    output = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                output.append(item)
    return output


def placement_label(row):
    enc = row.get("encoder_activation", row.get("activation", ""))
    dec = row.get("decoder_activation", row.get("activation", ""))
    out = row.get("output_activation", "linear")
    act = row.get("activation", "")
    if out != "linear" and enc == "gelu" and dec == "gelu":
        return "output-only"
    if out != "linear":
        return "hidden+output"
    if enc != "gelu" and dec == "gelu":
        return "encoder-only"
    if enc == "gelu" and dec != "gelu":
        return "decoder-only"
    if enc == dec == act:
        return "full-hidden"
    return "mixed-hidden"


def bin_series(series, factor_name, q):
    valid = pd.to_numeric(series, errors="coerce")
    labels = ["down", "flat", "up"] if factor_name == "trend" else ["low", "mid", "high"]
    if q != 3:
        labels = [f"q{i + 1}" for i in range(q)]
    if valid.dropna().nunique() < q:
        median = valid.median()
        return pd.Series(np.where(valid <= median, labels[0], labels[-1]), index=series.index)
    return pd.qcut(valid.rank(method="first"), q=q, labels=labels)


def available_sample_keys(df):
    return [column for column in SAMPLE_KEY_CANDIDATES if column in df.columns]


def dedupe_candidates(df, metric):
    required = {"activation_signature", metric}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    key_cols = available_sample_keys(df) + ["activation_signature"]
    df = df.copy()
    df["_row_order"] = np.arange(len(df))
    df = df.sort_values("_row_order").drop_duplicates(key_cols, keep="last")
    for column in ["activation", "encoder_activation", "decoder_activation", "output_activation"]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("")
    df.loc[df["encoder_activation"] == "", "encoder_activation"] = df.loc[
        df["encoder_activation"] == "", "activation"
    ]
    df.loc[df["decoder_activation"] == "", "decoder_activation"] = df.loc[
        df["decoder_activation"] == "", "activation"
    ]
    df.loc[df["output_activation"] == "", "output_activation"] = "linear"
    df["placement"] = df.apply(placement_label, axis=1)
    return df.drop(columns=["_row_order"])


def static_ranking(df, metric):
    group_cols = [
        "dataset",
        "pred_len",
        "activation",
        "encoder_activation",
        "decoder_activation",
        "output_activation",
        "activation_signature",
        "placement",
    ]
    group_cols = [column for column in group_cols if column in df.columns]
    ranking = (
        df.groupby(group_cols, dropna=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std", "count": "sample_count"})
    )
    if "seed" in df.columns:
        seed_counts = (
            df.groupby(group_cols, dropna=False)["seed"].nunique().reset_index().rename(columns={"seed": "seed_count"})
        )
        ranking = ranking.merge(seed_counts, on=group_cols, how="left")
    return ranking.sort_values(["dataset", "pred_len", f"{metric}_mean"])


def attach_factor_bins(df, factors, q):
    df = df.copy()
    sample_keys = available_sample_keys(df)
    unique_samples = df[sample_keys + [f"input_{factor}" for factor in factors]].drop_duplicates(sample_keys)
    for factor in factors:
        unique_samples[f"{factor}_bin"] = unique_samples.groupby(["dataset", "pred_len"], dropna=False)[
            f"input_{factor}"
        ].transform(lambda series: bin_series(series, factor, q))
    return df.merge(unique_samples[sample_keys + [f"{factor}_bin" for factor in factors]], on=sample_keys, how="left")


def best_static_by_cell(static, metric):
    value_col = f"{metric}_mean"
    static = static[pd.to_numeric(static[value_col], errors="coerce").notna()].copy()
    if static.empty:
        return pd.DataFrame(
            columns=["dataset", "pred_len", "best_static_signature", "best_static_placement", "best_static_mse"]
        )
    idx = static.groupby(["dataset", "pred_len"], dropna=False)[value_col].idxmin().dropna().astype(int)
    return static.loc[idx, ["dataset", "pred_len", "activation_signature", "placement", value_col]].rename(
        columns={
            "activation_signature": "best_static_signature",
            "placement": "best_static_placement",
            value_col: "best_static_mse",
        }
    )


def same_split_oracle(df, factors, metric):
    rows = []
    best_rows = []
    static = best_static_by_cell(static_ranking(df, metric), metric)
    for factor in factors:
        bin_col = f"{factor}_bin"
        group_cols = ["dataset", "pred_len", bin_col, "activation_signature", "placement"]
        ranked = (
            df.groupby(group_cols, dropna=False, observed=True)[metric]
            .mean()
            .reset_index()
            .rename(columns={metric: "bin_activation_mse"})
        )
        ranked = ranked[pd.to_numeric(ranked["bin_activation_mse"], errors="coerce").notna()].copy()
        if ranked.empty:
            continue
        idx = (
            ranked.groupby(["dataset", "pred_len", bin_col], dropna=False, observed=True)["bin_activation_mse"]
            .idxmin()
            .dropna()
            .astype(int)
        )
        best = ranked.loc[idx].copy()
        best["factor"] = factor
        best = best.rename(columns={bin_col: "factor_bin", "activation_signature": "oracle_signature"})
        best_rows.append(best)

        count_cols = list(dict.fromkeys(["dataset", "pred_len", bin_col] + available_sample_keys(df)))
        counts = df[count_cols].drop_duplicates()
        counts = counts.groupby(["dataset", "pred_len", bin_col], dropna=False, observed=True).size().reset_index(name="sample_count")
        weighted = best.merge(counts, left_on=["dataset", "pred_len", "factor_bin"], right_on=["dataset", "pred_len", bin_col])
        summary = (
            weighted.assign(weighted_mse=weighted["bin_activation_mse"] * weighted["sample_count"])
            .groupby(["dataset", "pred_len"], dropna=False)
            .agg(total_weighted_mse=("weighted_mse", "sum"), total_samples=("sample_count", "sum"), bin_count=("factor_bin", "nunique"))
            .reset_index()
        )
        summary["same_split_oracle_mse"] = summary["total_weighted_mse"] / summary["total_samples"]
        summary["factor"] = factor
        rows.append(summary.drop(columns=["total_weighted_mse"]))
    oracle = pd.concat(rows, ignore_index=True).merge(static, on=["dataset", "pred_len"], how="left")
    oracle["same_split_gain_vs_best_static"] = (
        oracle["best_static_mse"] - oracle["same_split_oracle_mse"]
    ) / oracle["best_static_mse"]
    return oracle, pd.concat(best_rows, ignore_index=True)


def leave_one_seed_out_oracle(df, factors, metric):
    if "seed" not in df.columns or df["seed"].nunique() < 2:
        return pd.DataFrame()
    rows = []
    static = best_static_by_cell(static_ranking(df, metric), metric)
    sample_keys = available_sample_keys(df)
    for factor in factors:
        bin_col = f"{factor}_bin"
        for (dataset, pred_len), cell in df.groupby(["dataset", "pred_len"], dropna=False):
            for seed in sorted(cell["seed"].dropna().unique()):
                train = cell[cell["seed"] != seed]
                eval_rows = cell[cell["seed"] == seed]
                if train.empty or eval_rows.empty:
                    continue
                ranked = (
                    train.groupby([bin_col, "activation_signature"], dropna=False, observed=True)[metric]
                    .mean()
                    .reset_index()
                )
                ranked = ranked[pd.to_numeric(ranked[metric], errors="coerce").notna()].copy()
                if ranked.empty:
                    continue
                idx = ranked.groupby(bin_col, dropna=False, observed=True)[metric].idxmin().dropna().astype(int)
                choices = ranked.loc[idx].rename(columns={"activation_signature": "chosen_signature", metric: "train_bin_mse"})
                eval_choice = eval_rows.merge(choices[[bin_col, "chosen_signature"]], on=bin_col, how="inner")
                eval_choice = eval_choice[eval_choice["activation_signature"] == eval_choice["chosen_signature"]]
                if eval_choice.empty:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "pred_len": pred_len,
                        "factor": factor,
                        "eval_seed": seed,
                        "loso_oracle_mse": eval_choice[metric].mean(),
                        "eval_sample_count": eval_choice[sample_keys].drop_duplicates().shape[0],
                    }
                )
    if not rows:
        return pd.DataFrame()
    loso = pd.DataFrame(rows)
    summary = (
        loso.groupby(["dataset", "pred_len", "factor"], dropna=False)
        .agg(loso_oracle_mse=("loso_oracle_mse", "mean"), eval_seed_count=("eval_seed", "nunique"))
        .reset_index()
        .merge(static, on=["dataset", "pred_len"], how="left")
    )
    summary["loso_gain_vs_best_static"] = (
        summary["best_static_mse"] - summary["loso_oracle_mse"]
    ) / summary["best_static_mse"]
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze regime-conditioned activation-placement oracle")
    parser.add_argument("sample_results_csv", help="phase1_factor_sample_results.csv from analyze_factor_effects.py")
    parser.add_argument("--output_dir", default="analysis_regime_oracle")
    parser.add_argument("--metric", default="sample_mse_target")
    parser.add_argument("--factor", action="append", default=[])
    parser.add_argument("--q", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.sample_results_csv)
    factors = split_values(args.factor) or DEFAULT_FACTORS
    factors = [factor for factor in factors if f"input_{factor}" in df.columns]
    if not factors:
        raise SystemExit("No requested input_* factor columns found")

    df = dedupe_candidates(df, args.metric)
    df = attach_factor_bins(df, factors, args.q)
    static = static_ranking(df, args.metric)
    oracle, bin_best = same_split_oracle(df, factors, args.metric)
    loso = leave_one_seed_out_oracle(df, factors, args.metric)

    static_path = os.path.join(args.output_dir, "static_activation_ranking.csv")
    oracle_path = os.path.join(args.output_dir, "factor_oracle_summary.csv")
    bin_path = os.path.join(args.output_dir, "factor_bin_best_activation.csv")
    loso_path = os.path.join(args.output_dir, "factor_oracle_loso_summary.csv")
    static.to_csv(static_path, index=False)
    oracle.to_csv(oracle_path, index=False)
    bin_best.to_csv(bin_path, index=False)
    loso.to_csv(loso_path, index=False)
    print(f"Wrote {static_path}: {len(static)} rows")
    print(f"Wrote {oracle_path}: {len(oracle)} rows")
    print(f"Wrote {bin_path}: {len(bin_best)} rows")
    print(f"Wrote {loso_path}: {len(loso)} rows")


if __name__ == "__main__":
    main()
