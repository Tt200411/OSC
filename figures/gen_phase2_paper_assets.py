#!/usr/bin/env python3
"""Generate Phase-2 paper tables and figures from the deduplicated aggregate CSV."""

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
AGG_PATH = ROOT / "phase2_remote_results" / "analysis_current" / "phase2_protocol_aggregate.csv"

GELU = "act=gelu|enc=gelu|dec=gelu|out=linear|a=0|lee=1"
TANH = "act=tanh|enc=tanh|dec=tanh|out=linear|a=0|lee=1"
SOFTSIGN = "act=softsign|enc=softsign|dec=softsign|out=linear|a=0|lee=1"
TANH_SIN = "act=tanh_sin|enc=tanh_sin|dec=tanh_sin|out=linear|a=0.01|lee=1"
ENC_TANH = "act=gelu|enc=tanh|dec=gelu|out=linear|a=0|lee=1"
DEC_TANH = "act=gelu|enc=gelu|dec=tanh|out=linear|a=0|lee=1"
TANH_OUT = "act=tanh|enc=tanh|dec=tanh|out=tanh|a=0|lee=1"
GELU_OUT = "act=gelu|enc=gelu|dec=gelu|out=tanh|a=0|lee=1"
LEE1 = "act=lee|enc=lee|dec=lee|out=linear|a=0|lee=1"
LEE3 = "act=lee|enc=lee|dec=lee|out=linear|a=0|lee=3"

METHODS = {
    GELU: "GELU",
    TANH: "tanh",
    SOFTSIGN: "softsign",
    TANH_SIN: r"tanh\_sin",
    ENC_TANH: "enc-tanh",
    DEC_TANH: "dec-tanh",
    TANH_OUT: "tanh+out",
    GELU_OUT: "gelu+out",
    LEE1: "Lee1",
    LEE3: "Lee3",
}


def escape_tex(value: object) -> str:
    return str(value).replace("_", r"\_")


def write_latex_table(path: Path, caption: str, label: str, columns: list[str], rows: list[list[str]], align: str) -> None:
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


def sync_to_paper(path: Path) -> None:
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    target = PAPER_FIG_DIR / path.name
    if path.resolve() != target.resolve():
        shutil.copy2(path, target)
        print(f"Copied {path.name} to paper/figures")


def load_agg() -> pd.DataFrame:
    if not AGG_PATH.exists():
        raise SystemExit(f"Missing aggregate CSV: {AGG_PATH}")
    agg = pd.read_csv(AGG_PATH)
    return agg


def row_for(agg: pd.DataFrame, dataset: str, pred_len: int, batch_size: int, signature: str) -> pd.Series | None:
    match = agg[
        (agg["dataset"] == dataset)
        & (agg["pred_len"] == pred_len)
        & (agg["batch_size"] == batch_size)
        & (agg["activation_signature"] == signature)
    ]
    if match.empty:
        return None
    return match.sort_values(["seed_count", "run_count", "mse_mean"], ascending=[False, False, True]).iloc[0]


def mse_cell(row: pd.Series | None, bold: bool = False) -> str:
    if row is None:
        return "--"
    mean = float(row["mse_mean"])
    std = row.get("mse_std", np.nan)
    if pd.isna(std):
        text = f"{mean:.3f}"
    else:
        text = rf"{mean:.3f} $\pm$ {float(std):.3f}"
    if bold:
        text = rf"\textbf{{{text}}}"
    return text


def gain_cell(row: pd.Series | None) -> str:
    if row is None or pd.isna(row.get("improvement_vs_gelu_mean", np.nan)):
        return "--"
    return f"{100.0 * float(row['improvement_vs_gelu_mean']):+.1f}\\%"


def seed_cell(row: pd.Series | None) -> str:
    if row is None:
        return "--"
    return str(int(row["seed_count"]))


def table_phase2_main_ett(agg: pd.DataFrame) -> None:
    cells = [("ETTh1", 168), ("ETTh1", 336), ("ETTh2", 168), ("ETTh2", 336)]
    method_sigs = [GELU, TANH, SOFTSIGN, TANH_SIN]
    rows = []
    for dataset, pred_len in cells:
        found = {sig: row_for(agg, dataset, pred_len, 32, sig) for sig in method_sigs}
        means = {sig: float(row["mse_mean"]) for sig, row in found.items() if row is not None}
        best_sig = min(means, key=means.get) if means else ""
        best_non_gelu = min((s for s in means if s != GELU), key=lambda s: means[s], default="")
        rows.append(
            [
                dataset,
                str(pred_len),
                mse_cell(found[GELU], GELU == best_sig),
                mse_cell(found[TANH], TANH == best_sig),
                mse_cell(found[SOFTSIGN], SOFTSIGN == best_sig),
                mse_cell(found[TANH_SIN], TANH_SIN == best_sig),
                gain_cell(found.get(best_non_gelu)),
                seed_cell(found[GELU]),
            ]
        )
    out = FIG_DIR / "TABLE_phase2_main_ett.tex"
    write_latex_table(
        out,
        "Phase-2 official-script Informer results on ETT long-horizon cells. Values are MSE mean $\\pm$ standard deviation over seeds; lower is better. Gain is the relative MSE reduction of the best non-GELU hidden activation against the same-protocol GELU baseline.",
        "tab:phase2_main_ett",
        ["Dataset", "H", "GELU", "tanh", "softsign", r"tanh\_sin", "best gain", "seeds"],
        rows,
        "llcccccc",
    )
    sync_to_paper(out)


def table_phase2_placement(agg: pd.DataFrame) -> None:
    cells = [
        ("ETTh1", 168),
        ("ETTh1", 336),
        ("ETTh2", 168),
        ("ETTh2", 336),
        ("Solar1", 96),
        ("Solar5", 96),
    ]
    method_sigs = [GELU, TANH, ENC_TANH, DEC_TANH, TANH_OUT]
    rows = []
    for dataset, pred_len in cells:
        found = {sig: row_for(agg, dataset, pred_len, 32, sig) for sig in method_sigs}
        means = {sig: float(row["mse_mean"]) for sig, row in found.items() if row is not None}
        best_sig = min(means, key=means.get) if means else ""
        rows.append(
            [
                dataset,
                str(pred_len),
                mse_cell(found[GELU], GELU == best_sig),
                mse_cell(found[TANH], TANH == best_sig),
                mse_cell(found[ENC_TANH], ENC_TANH == best_sig),
                mse_cell(found[DEC_TANH], DEC_TANH == best_sig),
                mse_cell(found[TANH_OUT], TANH_OUT == best_sig),
            ]
        )
    out = FIG_DIR / "TABLE_phase2_placement.tex"
    write_latex_table(
        out,
        "Placement and output-activation ablations under the Phase-2 protocol. Output tanh is not a universal source of gain: it helps selected ETT cells but is harmful on Solar96.",
        "tab:phase2_placement",
        ["Dataset", "H", "GELU", "tanh hidden", "enc-tanh", "dec-tanh", "tanh output"],
        rows,
        "llccccc",
    )
    sync_to_paper(out)


def table_phase2_lee_pilot(agg: pd.DataFrame) -> None:
    cells = [("ETTh1", 168, 8), ("ETTh2", 168, 4)]
    method_sigs = [GELU, TANH, SOFTSIGN, LEE1, LEE3]
    rows = []
    for dataset, pred_len, batch_size in cells:
        found = {sig: row_for(agg, dataset, pred_len, batch_size, sig) for sig in method_sigs}
        means = {sig: float(row["mse_mean"]) for sig, row in found.items() if row is not None}
        best_sig = min(means, key=means.get) if means else ""
        rows.append(
            [
                dataset,
                str(pred_len),
                str(batch_size),
                mse_cell(found[GELU], GELU == best_sig),
                mse_cell(found[TANH], TANH == best_sig),
                mse_cell(found[SOFTSIGN], SOFTSIGN == best_sig),
                mse_cell(found[LEE1], LEE1 == best_sig),
                mse_cell(found[LEE3], LEE3 == best_sig),
                ", ".join(seed_cell(found[sig]) for sig in method_sigs if found[sig] is not None),
            ]
        )
    out = FIG_DIR / "TABLE_phase2_lee_pilot.tex"
    write_latex_table(
        out,
        "Memory-adjusted Lee probes. These runs use smaller batches than the Phase-2 main table and are therefore diagnostic rather than main evidence.",
        "tab:phase2_lee_pilot",
        ["Dataset", "H", "batch", "GELU", "tanh", "softsign", "Lee1", "Lee3", "seed counts"],
        rows,
        "lllcccccc",
    )
    sync_to_paper(out)


def figure_relative_gains(agg: pd.DataFrame) -> None:
    cells = [
        ("ETTh1", 168),
        ("ETTh1", 336),
        ("ETTh2", 168),
        ("ETTh2", 336),
        ("Solar1", 96),
        ("Solar5", 96),
    ]
    methods = [(TANH, "tanh", "tanh"), (SOFTSIGN, "softsign", "softsign"), (TANH_SIN, r"tanh\_sin", "tanh_sin")]
    labels = [f"{dataset}-{pred_len}" for dataset, pred_len in cells]
    x = np.arange(len(cells))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for idx, (sig, label, color_key) in enumerate(methods):
        gains = []
        for dataset, pred_len in cells:
            row = row_for(agg, dataset, pred_len, 32, sig)
            gains.append(np.nan if row is None else 100.0 * float(row["improvement_vs_gelu_mean"]))
        ax.bar(x + (idx - 1) * width, gains, width=width, label=label, color=COLORS[color_key])
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("MSE reduction vs GELU (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(frameon=False, ncol=3)
    ax.set_title("Bounded hidden activations improve same-protocol GELU most strongly on ETT")
    fig.tight_layout()
    for suffix in ["pdf", "png"]:
        out = FIG_DIR / f"fig_phase2_relative_gains.{suffix}"
        save_fig(fig, out)
        sync_to_paper(out)
    plt.close(fig)


def latex_includes() -> None:
    text = r"""% Generated by figures/gen_phase2_paper_assets.py
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig_phase2_relative_gains.pdf}
    \caption{Relative MSE reduction against same-protocol GELU. Bounded hidden activations provide the clearest gains on ETT; Solar gains are smaller and more placement-sensitive.}
    \label{fig:phase2_relative_gains}
\end{figure}
"""
    out = FIG_DIR / "phase2_latex_includes.tex"
    out.write_text(text, encoding="utf-8")
    sync_to_paper(out)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    agg = load_agg()
    table_phase2_main_ett(agg)
    table_phase2_placement(agg)
    table_phase2_lee_pilot(agg)
    figure_relative_gains(agg)
    latex_includes()


if __name__ == "__main__":
    main()
