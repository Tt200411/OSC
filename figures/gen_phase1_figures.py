#!/usr/bin/env python3
"""Generate paper figures and tables from real Phase 1 result files."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from paper_plot_style import COLORS, save_fig


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
ARIS_DIR = ROOT / ".aris"


def pct(x):
    if pd.isna(x):
        return np.nan
    return 100.0 * float(x)


def cell_label(row):
    return f"{row['dataset']} / {int(row['pred_len'])}"


def activation_label(row):
    act = row["activation"]
    if act == "lee":
        return f"Lee{int(row['lee_type'])}"
    amp = float(row["amplitude"])
    if act in {"tanh_sin", "tanh_cos", "tanh_rand"}:
        return f"{act.replace('tanh_', '')} a={amp:g}"
    return act


def escape_tex(value):
    return str(value).replace("_", "\\_")


def write_latex_table(path, caption, label, columns, rows, align=None):
    align = align or ("l" * len(columns))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(columns) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(escape_tex(x) for x in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def load_tables():
    default_summary = pd.read_csv(ROOT / "phase1_remote_results/all_servers_current/phase1_activation_summary.csv")
    tanh_summary = pd.read_csv(ROOT / "phase1_remote_results/all_servers_current_tanh_baseline/phase1_activation_summary.csv")
    coverage = pd.read_csv(ARIS_DIR / "dataset_coverage_current.csv")
    factor_hl = pd.read_csv(ARIS_DIR / "factor_bin_high_low_lee_tanh.csv")
    return default_summary, tanh_summary, coverage, factor_hl


def fig_bounded_vs_perturbations(default_summary, tanh_summary):
    selected_cells = [
        ("ETTh1", 24),
        ("Solar1", 24),
        ("Solar1", 96),
        ("Solar5", 24),
        ("Solar5", 96),
    ]
    bounded = default_summary[
        (default_summary["seed_count"] >= 3)
        & (default_summary["activation"].isin(["tanh", "softsign", "lee"]))
        & (default_summary["dataset"].isin([x[0] for x in selected_cells]))
    ].copy()
    bounded = bounded[
        bounded.apply(lambda r: (r["dataset"], int(r["pred_len"])) in selected_cells, axis=1)
    ]
    bounded = bounded[(bounded["activation"] != "lee") | (bounded["lee_type"] == 1)]
    bounded["cell"] = bounded.apply(cell_label, axis=1)
    bounded["method"] = bounded.apply(activation_label, axis=1)
    bounded["change_pct"] = bounded["relative_mse_change_mean"].map(pct)
    bounded.to_csv(FIG_DIR / "fig1a_bounded_vs_gelu_data.csv", index=False)

    perturb = tanh_summary[
        (tanh_summary["seed_count"] >= 3)
        & (tanh_summary["activation"].isin(["tanh_sin", "tanh_cos", "tanh_rand"]))
    ].copy()
    perturb = perturb[
        perturb.apply(lambda r: (r["dataset"], int(r["pred_len"])) in [
            ("ETTh1", 168),
            ("ETTh2", 168),
            ("Solar1", 24),
            ("Solar1", 96),
            ("Solar5", 24),
            ("Solar5", 96),
        ], axis=1)
    ]
    perturb["cell"] = perturb.apply(cell_label, axis=1)
    perturb["method"] = perturb.apply(activation_label, axis=1)
    perturb["change_pct"] = perturb["relative_mse_change_mean"].map(pct)
    perturb.to_csv(FIG_DIR / "fig1b_perturb_vs_tanh_data.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.4))

    pivot = bounded.pivot_table(index="cell", columns="method", values="change_pct", aggfunc="mean")
    pivot = pivot.reindex([f"{d} / {p}" for d, p in selected_cells])
    methods = [m for m in ["tanh", "softsign", "Lee1"] if m in pivot.columns]
    x = np.arange(len(pivot.index))
    width = 0.24
    for i, method in enumerate(methods):
        axes[0].bar(
            x + (i - (len(methods) - 1) / 2) * width,
            pivot[method],
            width,
            label=method,
            color=COLORS.get(method.lower().replace("lee1", "lee"), "#888888"),
        )
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_ylabel("Relative MSE change vs GELU (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pivot.index, rotation=35, ha="right")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].text(0.0, 1.04, "(a) Bounded activations explain most GELU gains", transform=axes[0].transAxes, fontweight="bold")

    perturb = perturb.sort_values(["dataset", "pred_len", "activation", "amplitude"])
    plot_rows = perturb.copy()
    labels = [f"{cell_label(r)}\n{activation_label(r)}" for _, r in plot_rows.iterrows()]
    y = np.arange(len(plot_rows))
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in plot_rows["change_pct"]]
    axes[1].barh(y, plot_rows["change_pct"], color=colors)
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set_xlabel("Relative MSE change vs tanh (%)")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()
    axes[1].text(0.0, 1.04, "(b) Explicit perturbations are mixed vs tanh", transform=axes[1].transAxes, fontweight="bold")

    fig.tight_layout()
    save_fig(fig, FIG_DIR / "fig1_activation_effects.pdf")
    save_fig(fig, FIG_DIR / "fig1_activation_effects.png")
    plt.close(fig)


def fig_heatmap(tanh_summary):
    heat = tanh_summary[
        (tanh_summary["seed_count"] >= 3)
        & (tanh_summary["activation"].isin(["tanh_sin", "tanh_cos", "tanh_rand", "lee"]))
    ].copy()
    heat = heat[
        heat.apply(lambda r: (r["dataset"], int(r["pred_len"])) in [
            ("ETTh1", 24),
            ("ETTh1", 168),
            ("ETTh2", 168),
            ("Solar1", 24),
            ("Solar1", 96),
            ("Solar5", 24),
            ("Solar5", 96),
        ], axis=1)
    ]
    heat["cell"] = heat.apply(cell_label, axis=1)
    heat["method"] = heat.apply(activation_label, axis=1)
    heat["change_pct"] = heat["relative_mse_change_mean"].map(pct)
    methods = [
        "sin a=0.01",
        "sin a=0.05",
        "sin a=0.1",
        "cos a=0.01",
        "cos a=0.1",
        "rand a=0.01",
        "rand a=0.1",
        "Lee1",
        "Lee3",
    ]
    cells = ["ETTh1 / 24", "ETTh1 / 168", "ETTh2 / 168", "Solar1 / 24", "Solar1 / 96", "Solar5 / 24", "Solar5 / 96"]
    pivot = heat.pivot_table(index="method", columns="cell", values="change_pct", aggfunc="mean").reindex(methods).reindex(columns=cells)
    pivot.to_csv(FIG_DIR / "fig2_tanh_specificity_heatmap_data.csv")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    data = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    vmax = max(vmax, 5.0)
    im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(cells)))
    ax.set_xticklabels(cells, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if math.isfinite(value):
                ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Relative MSE change vs tanh (%)")
    ax.set_title("Lee and explicit perturbations are dataset- and horizon-dependent")
    fig.tight_layout()
    save_fig(fig, FIG_DIR / "fig2_tanh_specificity_heatmap.pdf")
    save_fig(fig, FIG_DIR / "fig2_tanh_specificity_heatmap.png")
    plt.close(fig)


def fig_factor_gradients(factor_hl):
    focus = factor_hl[
        (factor_hl["seed_count"] >= 3)
        & (factor_hl["factor"].isin(["volatility", "turbulence_score"]))
        & (factor_hl["activation"].isin(["tanh_sin", "lee"]))
    ].copy()
    focus["grad_pct"] = focus["relative_target_mse_change_mean_high_minus_low"].map(pct)
    focus = focus[np.isfinite(focus["grad_pct"])]
    focus = focus[
        focus.apply(lambda r: (r["dataset"], int(r["pred_len"])) in [
            ("ETTh1", 168),
            ("ETTh2", 168),
            ("Solar1", 24),
            ("Solar1", 96),
            ("Solar5", 24),
            ("Solar5", 96),
        ], axis=1)
    ]
    focus["label"] = focus.apply(
        lambda r: f"{r['dataset']} / {int(r['pred_len'])} / {activation_label(r)} / {r['factor']} / {r['source_host']}",
        axis=1,
    )
    focus = focus.reindex(focus["grad_pct"].abs().sort_values(ascending=False).index).head(16)
    focus = focus.sort_values("grad_pct")
    focus.to_csv(FIG_DIR / "fig3_factor_gradient_data.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in focus["grad_pct"]]
    y = np.arange(len(focus))
    ax.barh(y, focus["grad_pct"], color=colors)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(focus["label"])
    ax.set_xlabel("High-minus-low target relative MSE change (percentage points)")
    ax.set_title("Factor gradients expose conditional regimes, not universal wins")
    fig.tight_layout()
    save_fig(fig, FIG_DIR / "fig3_factor_gradients.pdf")
    save_fig(fig, FIG_DIR / "fig3_factor_gradients.png")
    plt.close(fig)


def write_tables(default_summary, tanh_summary, coverage, factor_hl):
    cov = coverage.copy()
    write_latex_table(
        FIG_DIR / "TABLE_dataset_coverage.tex",
        "Current synchronized experiment coverage. Low-seed rows are retained for provenance but are not used as strong claims.",
        "tab:dataset_coverage",
        ["Dataset", "Pred.", "Run rows", "Activations", "Seeds", "Servers"],
        [
            [r.dataset, int(r.pred_len), int(r.run_rows), int(r.activations), int(r.seeds), int(r.servers)]
            for _, r in cov.iterrows()
        ],
        align="lrrrrr",
    )

    bounded = default_summary[
        (default_summary["seed_count"] >= 3)
        & (default_summary["activation"].isin(["tanh", "softsign", "lee"]))
        & (default_summary["baseline_activation"] == "gelu")
        & default_summary["relative_mse_change_mean"].notna()
        & default_summary["win_rate"].notna()
    ].copy()
    bounded = bounded[(bounded["activation"] != "lee") | (bounded["lee_type"] == 1)]
    bounded["rel"] = bounded["relative_mse_change_mean"].map(lambda x: f"{100*x:+.2f}\\%" if pd.notna(x) else "--")
    bounded = bounded.sort_values(["dataset", "pred_len", "activation"])
    write_latex_table(
        FIG_DIR / "TABLE_bounded_vs_gelu.tex",
        "Activation results against GELU for bounded controls and Lee-OC rows with at least three seeds.",
        "tab:bounded_vs_gelu",
        ["Dataset", "Pred.", "Activation", "MSE", "MAE", "$\\Delta$MSE", "Win"],
        [
            [
                r.dataset,
                int(r.pred_len),
                activation_label(r),
                f"{r.mse_mean:.4f}",
                f"{r.mae_mean:.4f}",
                r.rel,
                f"{r.win_rate:.3f}" if pd.notna(r.win_rate) else "--",
            ]
            for _, r in bounded.iterrows()
        ],
        align="lrlrrrr",
    )

    spec = tanh_summary[
        (tanh_summary["seed_count"] >= 3)
        & (tanh_summary["activation"].isin(["tanh_sin", "tanh_cos", "tanh_rand", "lee"]))
    ].copy()
    spec["rel"] = spec["relative_mse_change_mean"].map(lambda x: f"{100*x:+.2f}\\%" if pd.notna(x) else "--")
    spec = spec.sort_values(["dataset", "pred_len", "activation", "amplitude", "lee_type"])
    write_latex_table(
        FIG_DIR / "TABLE_tanh_specificity.tex",
        "Specificity tests against the tanh baseline. Positive values indicate lower MSE than tanh.",
        "tab:tanh_specificity",
        ["Dataset", "Pred.", "Activation", "MSE", "$\\Delta$MSE", "Win", "Label"],
        [
            [
                r.dataset,
                int(r.pred_len),
                activation_label(r),
                f"{r.mse_mean:.4f}",
                r.rel,
                f"{r.win_rate:.3f}" if pd.notna(r.win_rate) else "--",
                r.effect_label,
            ]
            for _, r in spec.iterrows()
        ],
        align="lrlrrrr",
    )

    grad = factor_hl[
        (factor_hl["seed_count"] >= 3)
        & (factor_hl["activation"].isin(["tanh_sin", "lee"]))
        & (factor_hl["factor"].isin(["volatility", "turbulence_score"]))
    ].copy()
    grad["grad"] = grad["relative_target_mse_change_mean_high_minus_low"].map(
        lambda x: f"{100*x:+.2f} pp" if pd.notna(x) else "--"
    )
    grad["win_grad"] = grad["target_win_rate_high_minus_low"].map(
        lambda x: f"{100*x:+.1f} pp" if pd.notna(x) else "--"
    )
    grad = grad[np.isfinite(pd.to_numeric(grad["relative_target_mse_change_mean_high_minus_low"], errors="coerce"))]
    grad = grad.reindex(grad["relative_target_mse_change_mean_high_minus_low"].abs().sort_values(ascending=False).index).head(12)
    write_latex_table(
        FIG_DIR / "TABLE_factor_gradients.tex",
        "Largest current high-minus-low input-factor gradients for Lee and sinusoidal perturbations. These rows diagnose conditional regimes and do not by themselves imply global effectiveness.",
        "tab:factor_gradients",
        ["Dataset", "Pred.", "Activation", "Factor", "Server", "$\\Delta_{high-low}$", "Win shift"],
        [
            [
                r.dataset,
                int(r.pred_len),
                activation_label(r),
                r.factor,
                r.source_host,
                r.grad,
                r.win_grad,
            ]
            for _, r in grad.iterrows()
        ],
        align="lrlrrrr",
    )

    includes = FIG_DIR / "latex_includes.tex"
    includes.write_text(
        r"""% Generated by figures/gen_phase1_figures.py
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/fig1_activation_effects.pdf}
    \caption{Activation-only changes produce measurable Informer forecasting differences. (a) Bounded sign-preserving activations explain the strongest current gains over GELU. (b) Explicit perturbations around \texttt{tanh} are mixed and should not be treated as globally reliable.}
    \label{fig:activation_effects}
\end{figure*}

\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig2_tanh_specificity_heatmap.pdf}
    \caption{Specificity tests against \texttt{tanh}. Lee and explicit oscillatory perturbations are strongly dataset- and horizon-dependent.}
    \label{fig:tanh_specificity}
\end{figure}

\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{figures/fig3_factor_gradients.pdf}
    \caption{Input-factor gradients reveal conditional regimes. Positive values mean the high factor bin gains more than the low factor bin, but this diagnostic must be interpreted with global MSE and seed win rates.}
    \label{fig:factor_gradients}
\end{figure*}
""",
        encoding="utf-8",
    )
    print(f"Wrote {includes}")


def main():
    default_summary, tanh_summary, coverage, factor_hl = load_tables()
    fig_bounded_vs_perturbations(default_summary, tanh_summary)
    fig_heatmap(tanh_summary)
    fig_factor_gradients(factor_hl)
    write_tables(default_summary, tanh_summary, coverage, factor_hl)


if __name__ == "__main__":
    main()
