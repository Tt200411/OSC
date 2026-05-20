#!/usr/bin/env python3
"""Generate Phase-4 paper tables and figures from the full-grid aggregate CSVs."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from paper_plot_style import COLORS, save_fig


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
PAPER_FIG_DIR = ROOT / "paper" / "figures"
ANALYSIS_DIR = ROOT / "phase4_remote_results" / "analysis_current"
FACTOR_DIR = ANALYSIS_DIR / "factors"

AGG_PATH = ANALYSIS_DIR / "phase4_protocol_aggregate.csv"
COMPLETION_PATH = ANALYSIS_DIR / "phase4_completion_matrix.csv"
STATIC_RANK_PATH = FACTOR_DIR / "phase4_static_activation_ranking.csv"
LOSO_PATH = FACTOR_DIR / "phase4_factor_oracle_loso_summary.csv"


HORIZON_ORDER = [24, 48, 96, 168, 288, 336, 672, 720]
DATASET_ORDER = ["ETTh1", "ETTh2", "ETTm1", "ETTm2"] + [f"Solar{i}" for i in range(1, 9)]


DISPLAY_CONFIG = {
    "gelu_all": "GELU",
    "relu_all": "ReLU",
    "swish_all": "Swish",
    "tanh_all": "tanh",
    "softsign_all": "softsign",
    "tanh_sin001_all": r"tanh\_sin",
    "enc_tanh_dec_gelu": "enc-tanh",
    "enc_gelu_dec_tanh": "dec-tanh",
    "tanh_all_outtanh": r"tanh+out",
}

PLOT_CONFIG = {
    "gelu_all": "GELU",
    "relu_all": "ReLU",
    "swish_all": "Swish",
    "tanh_all": "tanh",
    "softsign_all": "softsign",
    "tanh_sin001_all": "tanh_sin",
    "enc_tanh_dec_gelu": "enc-tanh",
    "enc_gelu_dec_tanh": "dec-tanh",
    "tanh_all_outtanh": "tanh+out",
}


def escape_tex(value: object) -> str:
    return str(value).replace("_", r"\_")


def sync_to_paper(path: Path) -> None:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    target = PAPER_FIG_DIR / path.name
    if path.resolve() != target.resolve():
        shutil.copy2(path, target)
        print(f"Copied {path.name} to paper/figures")


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def write_table(path: Path, caption: str, label: str, columns: list[str], rows: list[list[str]], align: str) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")
    sync_to_paper(path)


def mse_std(row: pd.Series) -> str:
    return rf"{float(row['mse_mean']):.3f} $\pm$ {float(row['mse_std']):.3f}"


def pct(value: object, signed: bool = False) -> str:
    if pd.isna(value):
        return "--"
    fmt = "{:+.1f}\\%" if signed else "{:.1f}\\%"
    return fmt.format(100.0 * float(value))


def best_rows(agg: pd.DataFrame) -> pd.DataFrame:
    ordered = agg.sort_values(
        ["dataset", "pred_len", "mse_mean", "config_name"],
        ascending=[True, True, True, True],
    )
    return ordered.groupby(["dataset", "pred_len"], as_index=False).head(1)


def table_coverage(completion: pd.DataFrame) -> None:
    groups = [
        ("ETTh", completion["dataset"].isin(["ETTh1", "ETTh2"])),
        ("ETTm", completion["dataset"].isin(["ETTm1", "ETTm2"])),
        ("Solar", completion["dataset"].str.startswith("Solar")),
    ]
    rows = []
    for name, mask in groups:
        part = completion[mask]
        cells = part[["dataset", "pred_len"]].drop_duplicates().shape[0]
        config_cells = len(part)
        required = int(part["required_seed_runs"].sum())
        done = int(part["completed_seed_runs"].sum())
        complete = int(part["complete_3seeds"].sum())
        rows.append([name, str(cells), str(config_cells), f"{done}/{required}", f"{complete}/{config_cells}"])
    rows.append(
        [
            "All",
            str(completion[["dataset", "pred_len"]].drop_duplicates().shape[0]),
            str(len(completion)),
            f"{int(completion['completed_seed_runs'].sum())}/{int(completion['required_seed_runs'].sum())}",
            f"{int(completion['complete_3seeds'].sum())}/{len(completion)}",
        ]
    )
    write_table(
        FIG_DIR / "TABLE_phase4_coverage.tex",
        "Phase-4 full-grid coverage. A main-table cell is eligible only when all three seeds are present under batch size 32 and six training epochs.",
        "tab:phase4_coverage",
        ["Group", "dataset-horizon cells", "config cells", "seed-runs", "3-seed config cells"],
        rows,
        "lrrrr",
    )


def table_best_activation(agg: pd.DataFrame) -> None:
    best = best_rows(agg)
    best["dataset_order"] = best["dataset"].map({d: i for i, d in enumerate(DATASET_ORDER)})
    best["horizon_order"] = best["pred_len"].map({h: i for i, h in enumerate(HORIZON_ORDER)})
    best = best.sort_values(["dataset_order", "horizon_order"])
    rows = []
    for _, row in best.iterrows():
        rows.append(
            [
                escape_tex(row["dataset"]),
                str(int(row["pred_len"])),
                DISPLAY_CONFIG.get(str(row["config_name"]), escape_tex(row["config_name"])),
                mse_std(row),
                pct(row["improvement_vs_gelu_mean"], signed=True),
                pct(row["win_rate_vs_gelu"]),
            ]
        )
    write_table(
        FIG_DIR / "TABLE_phase4_best_activation.tex",
        "Best activation per dataset-horizon in the completed Phase-4 grid. Improvements are relative MSE reductions against the same-protocol GELU row with matched seeds; GELU-best cells have no relative-improvement entry.",
        "tab:phase4_best_activation",
        ["Dataset", "H", "best config", "MSE", "gain vs GELU", "seed win rate"],
        rows,
        "llcccc",
    )


def table_static_ranking(static_rank: pd.DataFrame) -> None:
    rows = []
    for _, row in static_rank.sort_values("mean_rank_in_cell").iterrows():
        rows.append(
            [
                DISPLAY_CONFIG.get(str(row["config_name"]), escape_tex(row["config_name"])),
                f"{float(row['mean_rank_in_cell']):.2f}",
                str(int(row["best_cell_count"])),
                str(int(row["top3_cell_count"])),
                pct(row["improvement_vs_gelu_mean"], signed=True),
                pct(row["positive_cell_rate"]),
            ]
        )
    write_table(
        FIG_DIR / "TABLE_phase4_static_ranking.tex",
        "Static activation ranking across the 36 completed dataset-horizon cells. Rank is computed within each cell over the nine Phase-4 configurations.",
        "tab:phase4_static_ranking",
        ["Config", "mean rank", "best cells", "top-3 cells", "mean gain", "positive cells"],
        rows,
        "lrrrrr",
    )


def table_loso_oracle(loso: pd.DataFrame) -> None:
    if loso.empty:
        rows = [["--", "--", "--", "--", "--", "--"]]
    else:
        positive = loso[loso["loso_gain_vs_best_static"] > 0].copy()
        positive = positive.sort_values("loso_gain_vs_best_static", ascending=False).head(8)
        rows = []
        for _, row in positive.iterrows():
            rows.append(
                [
                    escape_tex(row["dataset"]),
                    str(int(row["pred_len"])),
                    escape_tex(row["factor"]),
                    f"{float(row['loso_oracle_mse']):.3f}",
                    f"{float(row['best_static_mse']):.3f}",
                    pct(row["loso_gain_vs_best_static"], signed=True),
                ]
            )
    write_table(
        FIG_DIR / "TABLE_phase4_loso_oracle.tex",
        "Top positive leave-one-seed-out factor-oracle diagnostics among array-ready Phase-4 cells. These are diagnostic upper bounds for regime sensitivity, not deployed activation-routing results.",
        "tab:phase4_loso_oracle",
        ["Dataset", "H", "factor", "LOSO oracle MSE", "best static MSE", "gain"],
        rows,
        "lllccc",
    )


def fig_best_gain_heatmap(agg: pd.DataFrame) -> None:
    best = best_rows(agg)
    matrix = np.full((len(DATASET_ORDER), len(HORIZON_ORDER)), np.nan)
    labels = [["" for _ in HORIZON_ORDER] for _ in DATASET_ORDER]
    for _, row in best.iterrows():
        i = DATASET_ORDER.index(row["dataset"])
        j = HORIZON_ORDER.index(int(row["pred_len"]))
        gain = row["improvement_vs_gelu_mean"]
        matrix[i, j] = 0.0 if pd.isna(gain) else 100.0 * float(gain)
        labels[i][j] = PLOT_CONFIG.get(str(row["config_name"]), str(row["config_name"]).replace("_all", ""))

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad("#F0F0F0")
    im = ax.imshow(matrix, cmap=cmap, vmin=-20, vmax=65, aspect="auto")
    ax.set_xticks(np.arange(len(HORIZON_ORDER)))
    ax.set_xticklabels([str(h) for h in HORIZON_ORDER])
    ax.set_yticks(np.arange(len(DATASET_ORDER)))
    ax.set_yticklabels(DATASET_ORDER)
    ax.set_xlabel("Prediction horizon")
    ax.set_title("Best activation gain over same-protocol GELU")
    for i in range(len(DATASET_ORDER)):
        for j in range(len(HORIZON_ORDER)):
            if np.isnan(matrix[i, j]):
                continue
            color = "white" if matrix[i, j] > 35 or matrix[i, j] < -10 else "black"
            ax.text(j, i, f"{matrix[i, j]:.0f}%\n{labels[i][j]}", ha="center", va="center", fontsize=6.5, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("MSE reduction vs GELU (%)")
    fig.tight_layout()
    for suffix in ["pdf", "png"]:
        out = FIG_DIR / f"fig_phase4_best_gain_heatmap.{suffix}"
        save_fig(fig, out)
        sync_to_paper(out)
    plt.close(fig)


def fig_activation_ranking(static_rank: pd.DataFrame) -> None:
    rank = static_rank.sort_values("mean_rank_in_cell", ascending=True)
    labels = [PLOT_CONFIG.get(str(v), str(v).replace("_all", "")) for v in rank["config_name"]]
    x = np.arange(len(rank))
    fig, ax1 = plt.subplots(figsize=(7.0, 3.2))
    bars = ax1.bar(x, rank["best_cell_count"], color=COLORS["tanh"], alpha=0.85, label="best cells")
    ax1.set_ylabel("Best-cell count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, rank["mean_rank_in_cell"], color=COLORS["gelu"], marker="o", linewidth=1.3, label="mean rank")
    ax2.invert_yaxis()
    ax2.set_ylabel("Mean rank (lower is better)")
    for bar, value in zip(bars, rank["best_cell_count"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, str(int(value)), ha="center", va="bottom", fontsize=7)
    ax1.set_title("No single activation dominates the full grid")
    fig.tight_layout()
    for suffix in ["pdf", "png"]:
        out = FIG_DIR / f"fig_phase4_static_ranking.{suffix}"
        save_fig(fig, out)
        sync_to_paper(out)
    plt.close(fig)


def latex_includes() -> None:
    text = r"""% Generated by figures/gen_phase4_paper_assets.py
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/fig_phase4_best_gain_heatmap.pdf}
    \caption{Best activation gain over the same-protocol GELU row in each completed dataset-horizon cell. The full grid covers 36 dataset-horizon cells, nine activation configurations, and three seeds. Most cells prefer a non-GELU activation, but Solar3 remains GELU-best at both horizons.}
    \label{fig:phase4_best_gain_heatmap}
\end{figure*}

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig_phase4_static_ranking.pdf}
    \caption{Static activation ranking across the full Phase-4 grid. Bounded and tanh-family activations rank best on average, but the best-cell counts are spread across several configurations, arguing against a universal replacement activation.}
    \label{fig:phase4_static_ranking}
\end{figure}
"""
    out = FIG_DIR / "phase4_latex_includes.tex"
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}")
    sync_to_paper(out)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    agg = load_csv(AGG_PATH)
    completion = load_csv(COMPLETION_PATH)
    static_rank = load_csv(STATIC_RANK_PATH)
    loso = load_csv(LOSO_PATH)
    table_coverage(completion)
    table_best_activation(agg)
    table_static_ranking(static_rank)
    table_loso_oracle(loso)
    fig_best_gain_heatmap(agg)
    fig_activation_ranking(static_rank)
    latex_includes()


if __name__ == "__main__":
    main()
