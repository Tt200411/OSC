#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from phase4_report_utils import markdown_table
from phase4_router_feasibility import build_feature_table, safe_corr


ACTION_ORDER = ["baseline", "bounded_linear", "output_tanh", "other_linear"]


def ensure_columns(df, defaults):
    out = df.copy()
    for column, value in defaults.items():
        if column not in out.columns:
            out[column] = value
        out[column] = out[column].fillna(value)
    return out


def load_scope(path, model_scope, model_family, baseline_config, search_space):
    df = pd.read_csv(path)
    defaults = {
        "label_len": 0,
        "d_layers": 0,
        "factor": 0,
        "patch_len": 0,
        "stride": 0,
        "d_model": 0,
        "n_heads": 0,
        "d_ff": 0,
        "batch_size": 0,
        "train_epochs": 0,
        "placement": "",
    }
    df = ensure_columns(df, defaults)
    if "mse_mean" not in df.columns and "mse" in df.columns:
        group_cols = [
            "dataset",
            "pred_len",
            "seq_len",
            "label_len",
            "e_layers",
            "d_layers",
            "factor",
            "d_model",
            "n_heads",
            "d_ff",
            "batch_size",
            "train_epochs",
            "patch_len",
            "stride",
            "config_name",
            "placement",
        ]
        group_cols = [col for col in group_cols if col in df.columns]
        df = df.groupby(group_cols, dropna=False).agg(mse_mean=("mse", "mean")).reset_index()
        df = ensure_columns(df, defaults)
    df["model_scope"] = model_scope
    df["model_family"] = model_family
    df["baseline_config"] = baseline_config
    df["search_space"] = search_space
    return df


def config_action(config_name, baseline_config):
    name = str(config_name)
    if name == str(baseline_config):
        return "baseline"
    if "outtanh" in name or "out=tanh" in name:
        return "output_tanh"
    if "tanh" in name or "softsign" in name:
        return "bounded_linear"
    return "other_linear"


def model_metadata(rows):
    out = rows.copy()
    out["pred_to_seq_ratio"] = out["pred_len"] / out["seq_len"].replace(0, np.nan)
    out["ffn_to_model_ratio"] = out["d_ff"] / out["d_model"].replace(0, np.nan)
    out["has_patch"] = (out["patch_len"] > 0).astype(float)
    out["patch_to_seq_ratio"] = out["patch_len"] / out["seq_len"].replace(0, np.nan)
    out["stride_to_patch_ratio"] = out["stride"] / out["patch_len"].replace(0, np.nan)
    out["has_decoder"] = (out["d_layers"] > 0).astype(float)
    out["is_informer"] = (out["model_family"] == "Informer").astype(float)
    out["is_patchtst"] = (out["model_family"] == "PatchTST").astype(float)
    out["is_itransformer"] = (out["model_family"] == "iTransformer").astype(float)
    if "config_name" in out.columns:
        output_tanh_source = out["config_name"].astype(str)
    else:
        config_cols = [c for c in ["oracle_config", "static_config", "search_space"] if c in out.columns]
        output_tanh_source = out[config_cols].astype(str).agg(" ".join, axis=1) if config_cols else pd.Series("", index=out.index)
    if "scope_has_output_tanh" not in out.columns:
        scope_has_outtanh = output_tanh_source.groupby(out["model_scope"]).transform(
            lambda s: float(s.astype(str).str.contains("outtanh|out=tanh|placement_probe", regex=True).any())
        )
        out["scope_has_output_tanh"] = scope_has_outtanh
    if "scope_config_count" not in out.columns:
        if "config_name" in out.columns:
            out["scope_config_count"] = out.groupby("model_scope")["config_name"].transform("nunique").astype(float)
        else:
            out["scope_config_count"] = out.groupby("model_scope")["search_space"].transform("count").astype(float)
    return out


def build_model_cells(all_results):
    df = all_results.copy()
    df["action"] = df.apply(lambda row: config_action(row["config_name"], row["baseline_config"]), axis=1)
    scope_stats = df.groupby("model_scope", dropna=False).agg(
        scope_config_count=("config_name", "nunique"),
        scope_has_output_tanh=("config_name", lambda s: float(s.astype(str).str.contains("outtanh|out=tanh", regex=True).any())),
    ).reset_index()
    keys = [
        "model_scope",
        "model_family",
        "search_space",
        "dataset",
        "pred_len",
        "seq_len",
        "label_len",
        "e_layers",
        "d_layers",
        "factor",
        "d_model",
        "n_heads",
        "d_ff",
        "batch_size",
        "train_epochs",
        "patch_len",
        "stride",
        "baseline_config",
    ]
    idx = df.groupby(keys, dropna=False)["mse_mean"].idxmin()
    best = df.loc[idx, keys + ["config_name", "mse_mean", "action"]].rename(
        columns={"config_name": "oracle_config", "mse_mean": "oracle_mse", "action": "oracle_action"}
    )
    base = df[df["config_name"] == df["baseline_config"]][keys + ["mse_mean"]].rename(
        columns={"mse_mean": "baseline_mse"}
    )
    cells = best.merge(base, on=keys, how="left")

    ranked = df.copy()
    ranked["rank_in_cell"] = ranked.groupby(keys, dropna=False)["mse_mean"].rank(method="min")
    static = ranked.groupby(["model_scope", "config_name"], dropna=False)["rank_in_cell"].mean().reset_index()
    static = static.sort_values(["model_scope", "rank_in_cell", "config_name"]).groupby("model_scope").head(1)
    static = static.rename(columns={"config_name": "static_config"})
    static_mse = df.merge(static[["model_scope", "static_config"]], on="model_scope", how="left")
    static_mse = static_mse[static_mse["config_name"] == static_mse["static_config"]][keys + ["static_config", "mse_mean"]].rename(
        columns={"mse_mean": "static_mse"}
    )
    cells = cells.merge(static_mse, on=keys, how="left")
    cells["oracle_gain_vs_baseline"] = (cells["baseline_mse"] - cells["oracle_mse"]) / cells["baseline_mse"]
    cells["static_gain_vs_baseline"] = (cells["baseline_mse"] - cells["static_mse"]) / cells["baseline_mse"]
    cells["static_regret_vs_oracle"] = (cells["static_mse"] - cells["oracle_mse"]) / cells["oracle_mse"]
    cells["baseline_is_oracle"] = cells["oracle_action"] == "baseline"
    cells = cells.merge(scope_stats, on="model_scope", how="left")
    return cells, df


def build_action_table(results, cells):
    keys = [
        "model_scope",
        "dataset",
        "pred_len",
        "seq_len",
        "label_len",
        "e_layers",
        "d_layers",
        "factor",
    ]
    rows = []
    for cell_key, group in results.groupby(keys, dropna=False):
        cell = dict(zip(keys, cell_key))
        scope = group["model_scope"].iloc[0]
        baseline_config = group["baseline_config"].iloc[0]
        for action, action_group in group.groupby("action", dropna=False):
            best_row = action_group.sort_values("mse_mean").iloc[0]
            rows.append(
                {
                    **cell,
                    "action": action,
                    "action_config": best_row["config_name"],
                    "action_mse": float(best_row["mse_mean"]),
                    "model_scope": scope,
                    "baseline_config": baseline_config,
                }
            )
    out = pd.DataFrame(rows)
    cell_scores = cells[
        keys
        + [
            "baseline_mse",
            "oracle_mse",
            "static_mse",
            "oracle_action",
            "static_config",
        ]
    ].drop_duplicates()
    out = out.merge(cell_scores, on=keys, how="left")
    out["action_gain_vs_baseline"] = (out["baseline_mse"] - out["action_mse"]) / out["baseline_mse"]
    out["action_regret_vs_oracle"] = (out["action_mse"] - out["oracle_mse"]) / out["oracle_mse"]
    out["action_gain_vs_static"] = (out["static_mse"] - out["action_mse"]) / out["static_mse"]
    return out


def standardize_feature_frame(cells, factors_dir, lee_root):
    factor_agg = cells[
        ["dataset", "pred_len", "seq_len", "label_len", "e_layers", "d_layers", "factor"]
    ].drop_duplicates()
    features = build_feature_table(factor_agg, factors_dir, lee_root, max_windows=1024)
    merge_keys = ["dataset", "pred_len", "seq_len", "label_len", "e_layers", "d_layers", "factor"]
    out = cells.merge(features, on=merge_keys, how="left")
    out = model_metadata(out)
    return out


def usable_features(df):
    prefixes = ("input_", "router_")
    meta = [
        "pred_len",
        "seq_len",
        "pred_to_seq_ratio",
        "e_layers",
        "d_layers",
        "d_model",
        "n_heads",
        "d_ff",
        "ffn_to_model_ratio",
        "batch_size",
        "train_epochs",
        "patch_len",
        "stride",
        "patch_to_seq_ratio",
        "stride_to_patch_ratio",
        "has_patch",
        "has_decoder",
        "is_informer",
        "is_patchtst",
        "is_itransformer",
        "scope_has_output_tanh",
        "scope_config_count",
    ]
    cols = [c for c in df.columns if c.startswith(prefixes) and not c.endswith("_bin")]
    cols += [c for c in meta if c in df.columns]
    usable = []
    for col in dict.fromkeys(cols):
        if col in {"router_source_csv", "router_window_count"}:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() >= 20 and values.nunique(dropna=True) >= 2:
            df[col] = values
            usable.append(col)
    return usable


def split_groups(df, mode):
    reset = df.reset_index(drop=True)
    if mode == "leave_cell":
        return [
            (f"{r.model_scope}:{r.dataset}/{int(r.pred_len)}", reset.index[
                (reset["model_scope"] == r.model_scope)
                & (reset["dataset"] == r.dataset)
                & (reset["pred_len"] == r.pred_len)
            ].tolist())
            for r in reset[["model_scope", "dataset", "pred_len"]].drop_duplicates().itertuples(index=False)
        ]
    if mode == "leave_dataset":
        return [(d, reset.index[reset["dataset"] == d].tolist()) for d in sorted(reset["dataset"].unique())]
    if mode == "leave_model_scope":
        return [(m, reset.index[reset["model_scope"] == m].tolist()) for m in sorted(reset["model_scope"].unique())]
    if mode == "leave_family":
        return [(m, reset.index[reset["model_family"] == m].tolist()) for m in sorted(reset["model_family"].unique())]
    raise ValueError(mode)


def bin_value(value, q1, q2):
    if pd.isna(value):
        return "mid"
    if value <= q1:
        return "low"
    if value <= q2:
        return "mid"
    return "high"


def action_allowed_for_scope(train_actions, scope, action):
    return not train_actions[(train_actions["model_scope"] == scope) & (train_actions["action"] == action)].empty


def fallback_action_for_scope(train_actions, scope):
    scope_rows = train_actions[train_actions["model_scope"] == scope]
    if scope_rows.empty:
        return "baseline"
    score_col = "action_regret_vs_oracle" if "action_regret_vs_oracle" in scope_rows.columns else "action_mse"
    means = scope_rows.groupby("action")[score_col].mean().sort_values()
    return str(means.index[0])


def fit_rule(train_cells, train_actions, feature):
    values = pd.to_numeric(train_cells[feature], errors="coerce").dropna()
    if values.nunique() < 3:
        q1 = q2 = float(values.median()) if not values.empty else 0.0
    else:
        q1, q2 = np.nanquantile(values.to_numpy(float), [1 / 3, 2 / 3])
    feature_frame = train_cells[["model_scope", "dataset", "pred_len"]].copy()
    feature_frame["__feature_value"] = train_cells[feature].to_numpy()
    train = train_actions.merge(
        feature_frame,
        on=["model_scope", "dataset", "pred_len"],
        how="inner",
    )
    train["factor_bin"] = train["__feature_value"].apply(lambda v: bin_value(v, q1, q2))
    choices = {}
    for (scope, factor_bin), group in train.groupby(["model_scope", "factor_bin"], dropna=False):
        score_col = "action_regret_vs_oracle" if "action_regret_vs_oracle" in group.columns else "action_mse"
        scores = group.groupby("action")[score_col].mean().sort_values()
        choices[(scope, str(factor_bin))] = str(scores.index[0]) if not scores.empty else "baseline"
    return {"feature": feature, "q1": float(q1), "q2": float(q2), "choices": choices}


def choose_action(rule, row, train_actions):
    scope = row["model_scope"]
    factor_bin = bin_value(row.get(rule["feature"], np.nan), rule["q1"], rule["q2"])
    action = rule["choices"].get((scope, factor_bin))
    if action is None:
        action = fallback_action_for_scope(train_actions, scope)
    if not action_allowed_for_scope(train_actions, scope, action):
        action = fallback_action_for_scope(train_actions, scope)
    return action


def action_to_config(train_actions, scope, action):
    group = train_actions[(train_actions["model_scope"] == scope) & (train_actions["action"] == action)]
    if group.empty:
        group = train_actions[train_actions["model_scope"] == scope]
    if group.empty:
        return None
    score_col = "action_regret_vs_oracle" if "action_regret_vs_oracle" in group.columns else "action_mse"
    means = group.groupby(["action_config"], dropna=False)[score_col].mean().sort_values()
    return str(means.index[0])


def first_lookup(lookup, key):
    row = lookup.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


def evaluate_rule(rule, test_cells, train_actions, all_actions, all_results, evaluation_granularity):
    action_lookup = all_actions.set_index(["model_scope", "dataset", "pred_len", "action"])
    config_lookup = all_results.set_index(["model_scope", "dataset", "pred_len", "config_name"])
    rows = []
    for _, row in test_cells.iterrows():
        action = choose_action(rule, row, train_actions)
        config = action_to_config(train_actions, row["model_scope"], action)
        fallback_used = False
        if evaluation_granularity == "action_oracle":
            try:
                action_row = first_lookup(action_lookup, (row["model_scope"], row["dataset"], row["pred_len"], action))
                chosen_mse = float(action_row["action_mse"])
                config = str(action_row["action_config"])
            except KeyError:
                action = "baseline"
                action_row = first_lookup(action_lookup, (row["model_scope"], row["dataset"], row["pred_len"], action))
                chosen_mse = float(action_row["action_mse"])
                config = str(action_row["action_config"])
                fallback_used = True
        else:
            if config is None or str(config) == "nan":
                config = row["baseline_config"]
                action = "baseline"
                fallback_used = True
            try:
                config_row = first_lookup(config_lookup, (row["model_scope"], row["dataset"], row["pred_len"], str(config)))
                chosen_mse = float(config_row["mse_mean"])
            except KeyError:
                config = row["baseline_config"]
                action = "baseline"
                config_row = first_lookup(config_lookup, (row["model_scope"], row["dataset"], row["pred_len"], str(config)))
                chosen_mse = float(config_row["mse_mean"])
                fallback_used = True
        rows.append(
            {
                "model_scope": row["model_scope"],
                "model_family": row["model_family"],
                "dataset": row["dataset"],
                "pred_len": row["pred_len"],
                "chosen_action": action,
                "chosen_config": config,
                "chosen_mse": chosen_mse,
                "oracle_action": row["oracle_action"],
                "oracle_config": row["oracle_config"],
                "oracle_mse": row["oracle_mse"],
                "baseline_mse": row["baseline_mse"],
                "static_config": row["static_config"],
                "static_mse": row["static_mse"],
                "fallback_used": fallback_used,
            }
        )
    return score_predictions(pd.DataFrame(rows))


def score_predictions(df):
    out = df.copy()
    out["gain_vs_baseline"] = (out["baseline_mse"] - out["chosen_mse"]) / out["baseline_mse"]
    out["gain_vs_static"] = (out["static_mse"] - out["chosen_mse"]) / out["static_mse"]
    out["regret_vs_oracle"] = (out["chosen_mse"] - out["oracle_mse"]) / out["oracle_mse"]
    out["action_hit"] = out["chosen_action"] == out["oracle_action"]
    out["config_hit"] = out["chosen_config"] == out["oracle_config"]
    return out


def inner_select_feature(train_cells, all_actions, all_results, feature_cols, evaluation_granularity):
    if len(train_cells) < 10:
        return feature_cols[0], pd.DataFrame()
    rows = []
    reset = train_cells.reset_index(drop=True)
    for feature in feature_cols:
        predictions = []
        for _, test_idx in split_groups(reset, "leave_cell"):
            test = reset.loc[test_idx]
            train = reset.drop(index=test_idx)
            train_keys = train[["model_scope", "dataset", "pred_len"]]
            train_actions = all_actions.merge(train_keys, on=["model_scope", "dataset", "pred_len"], how="inner")
            rule = fit_rule(train, train_actions, feature)
            predictions.append(evaluate_rule(rule, test, train_actions, all_actions, all_results, evaluation_granularity))
        pred = pd.concat(predictions, ignore_index=True)
        rows.append(
            {
                "feature": feature,
                "inner_mean_gain_vs_static": pred["gain_vs_static"].mean(),
                "inner_positive_vs_static_rate": (pred["gain_vs_static"] > 0).mean(),
                "inner_regret_vs_oracle": pred["regret_vs_oracle"].mean(),
                "inner_action_hit_rate": pred["action_hit"].mean(),
            }
        )
    scores = pd.DataFrame(rows).sort_values(
        ["inner_mean_gain_vs_static", "inner_positive_vs_static_rate", "inner_regret_vs_oracle"],
        ascending=[False, False, True],
    )
    return str(scores.iloc[0]["feature"]), scores


def train_score_select_feature(train_cells, all_actions, all_results, feature_cols, evaluation_granularity):
    if len(train_cells) < 4:
        return feature_cols[0], pd.DataFrame()
    train_keys = train_cells[["model_scope", "dataset", "pred_len"]]
    train_actions = all_actions.merge(train_keys, on=["model_scope", "dataset", "pred_len"], how="inner")
    rows = []
    for feature in feature_cols:
        rule = fit_rule(train_cells, train_actions, feature)
        pred = evaluate_rule(rule, train_cells, train_actions, all_actions, all_results, evaluation_granularity)
        rows.append(
            {
                "feature": feature,
                "train_mean_gain_vs_static": pred["gain_vs_static"].mean(),
                "train_positive_vs_static_rate": (pred["gain_vs_static"] > 0).mean(),
                "train_regret_vs_oracle": pred["regret_vs_oracle"].mean(),
                "train_action_hit_rate": pred["action_hit"].mean(),
            }
        )
    scores = pd.DataFrame(rows).sort_values(
        ["train_mean_gain_vs_static", "train_positive_vs_static_rate", "train_regret_vs_oracle"],
        ascending=[False, False, True],
    )
    return str(scores.iloc[0]["feature"]), scores


def evaluate_router(cells, actions, results, feature_cols, mode, selection_mode, evaluation_granularity):
    predictions = []
    inner_scores = []
    reset = cells.reset_index(drop=True)
    for fold, test_idx in split_groups(reset, mode):
        test = reset.loc[test_idx]
        train = reset.drop(index=test_idx)
        if train.empty or test.empty:
            continue
        if selection_mode == "nested":
            feature, scores = inner_select_feature(train, actions, results, feature_cols, evaluation_granularity)
        else:
            feature, scores = train_score_select_feature(train, actions, results, feature_cols, evaluation_granularity)
        scores["outer_mode"] = mode
        scores["outer_fold"] = str(fold)
        scores["selection_mode"] = selection_mode
        inner_scores.append(scores)
        train_keys = train[["model_scope", "dataset", "pred_len"]]
        train_actions = actions.merge(train_keys, on=["model_scope", "dataset", "pred_len"], how="inner")
        rule = fit_rule(train, train_actions, feature)
        pred = evaluate_rule(rule, test, train_actions, actions, results, evaluation_granularity)
        pred["split_mode"] = mode
        pred["fold"] = str(fold)
        pred["selected_feature"] = feature
        predictions.append(pred)
    return pd.concat(predictions, ignore_index=True), pd.concat(inner_scores, ignore_index=True)


def summarize(name, pred, static_regret):
    return {
        "eval_name": name,
        "cell_count": int(len(pred)),
        "mean_gain_vs_baseline": float(pred["gain_vs_baseline"].mean()),
        "mean_gain_vs_static": float(pred["gain_vs_static"].mean()),
        "positive_vs_static_rate": float((pred["gain_vs_static"] > 0).mean()),
        "mean_regret_vs_oracle": float(pred["regret_vs_oracle"].mean()),
        "regret_reduction_vs_static": float(
            (static_regret - pred["regret_vs_oracle"].mean()) / static_regret
        )
        if static_regret > 1e-8
        else np.nan,
        "action_hit_rate": float(pred["action_hit"].mean()),
        "config_hit_rate": float(pred["config_hit"].mean()),
    }


def correlation_table(cells, feature_cols):
    rows = []
    targets = ["oracle_gain_vs_baseline", "static_regret_vs_oracle"]
    cells["oracle_nonbaseline"] = (~cells["baseline_is_oracle"]).astype(float)
    targets.append("oracle_nonbaseline")
    for target in targets:
        for feature in feature_cols:
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "spearman": safe_corr(cells[feature], cells[target], "spearman"),
                    "pearson": safe_corr(cells[feature], cells[target], "pearson"),
                }
            )
    out = pd.DataFrame(rows)
    out["abs_spearman"] = out["spearman"].abs()
    return out.sort_values("abs_spearman", ascending=False)


def write_summary(path, context):
    lines = [
        "# Model-Aware Router Feasibility Summary",
        "",
        f"- Model-cell rows: {context['cell_count']}",
        f"- Feature count: {context['feature_count']}",
        f"- Router feature search count: {context['router_feature_count']}",
        f"- Static mean regret vs oracle: {context['static_regret']:.4f}",
        f"- Feature-selection mode: {context['selection_mode']}",
        f"- Evaluation granularity: {context['evaluation_granularity']}",
        "",
        "## Evaluation",
        "",
        markdown_table(context["eval_summary"]),
        "",
        "## Top Correlations",
        "",
        markdown_table(context["correlations"].head(15)),
        "",
        "## Verdict",
        "",
        context["verdict"],
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Cross-architecture model-aware activation router audit")
    parser.add_argument("--factors_dir", required=True)
    parser.add_argument("--output_dir", default="phase4_remote_results/analysis_current/factors_router/model_aware")
    parser.add_argument("--lee_root", default="lee_ocil")
    parser.add_argument("--router_top_features", type=int, default=20)
    parser.add_argument("--informer", default="phase4_remote_results/analysis_current/phase4_protocol_aggregate.csv")
    parser.add_argument("--patchtst", default="patchtst_remote_results/analysis_patchtst_transfer_actfix/patchtst_transfer_aggregate.csv")
    parser.add_argument("--itransformer_stage1", default="itransformer_remote_results/analysis_itransformer_stage1_actfix3/itransformer_protocol_aggregate.csv")
    parser.add_argument("--itransformer_placeprobe", default="itransformer_remote_results/analysis_itransformer_placeprobe_20260522/itransformer_protocol_aggregate.csv")
    parser.add_argument("--selection_mode", choices=["train_score", "nested"], default="train_score")
    parser.add_argument("--evaluation_granularity", choices=["config", "action_oracle"], default="config")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    scopes = [
        load_scope(args.informer, "Informer-fullgrid", "Informer", "gelu_all", "fullgrid"),
        load_scope(args.patchtst, "PatchTST-transfer", "PatchTST", "gelu_ffn_linear", "transfer"),
        load_scope(args.itransformer_stage1, "iTransformer-stage1", "iTransformer", "gelu_ffn_linear", "ffn_only"),
        load_scope(args.itransformer_placeprobe, "iTransformer-placeprobe", "iTransformer", "gelu_ffn_linear", "placement_probe"),
    ]
    results = pd.concat(scopes, ignore_index=True)
    cells, results_with_actions = build_model_cells(results)
    actions = build_action_table(results_with_actions, cells)
    cells = standardize_feature_frame(cells, args.factors_dir, args.lee_root)
    feature_cols = usable_features(cells)
    corr = correlation_table(cells, feature_cols)
    router_features = corr["feature"].drop_duplicates().head(args.router_top_features).tolist()
    if len(router_features) < 5:
        router_features = feature_cols

    evals = []
    inners = []
    for mode in ["leave_cell", "leave_dataset", "leave_model_scope", "leave_family"]:
        pred, inner = evaluate_router(
            cells,
            actions,
            results_with_actions,
            router_features,
            mode,
            args.selection_mode,
            args.evaluation_granularity,
        )
        pred.to_csv(os.path.join(args.output_dir, f"phase4_model_aware_router_{mode}.csv"), index=False)
        inner.to_csv(os.path.join(args.output_dir, f"phase4_model_aware_router_{mode}_inner_scores.csv"), index=False)
        evals.append(summarize(mode, pred, float(cells["static_regret_vs_oracle"].mean())))
        inners.append(inner)
    eval_summary = pd.DataFrame(evals)

    go_row = eval_summary[eval_summary["eval_name"] == "leave_cell"].iloc[0]
    dataset_row = eval_summary[eval_summary["eval_name"] == "leave_dataset"].iloc[0]
    if (
        go_row["mean_gain_vs_static"] > 0.005
        and go_row["positive_vs_static_rate"] >= 0.45
        and dataset_row["mean_gain_vs_static"] > -0.0025
    ):
        verdict = "PARTIAL-GO: model-aware factors improve the framing beyond Informer-only routing. Use as preliminary evidence, not a final deployable router, until more balanced PatchTST/iTransformer seeds/cells are added."
    else:
        verdict = "NO-GO: even with model metadata, the current small cross-architecture matrix is not strong enough for a deployable router claim."

    cells.to_csv(os.path.join(args.output_dir, "phase4_model_aware_router_cells.csv"), index=False)
    actions.to_csv(os.path.join(args.output_dir, "phase4_model_aware_router_actions.csv"), index=False)
    results_with_actions.to_csv(os.path.join(args.output_dir, "phase4_model_aware_router_results.csv"), index=False)
    corr.to_csv(os.path.join(args.output_dir, "phase4_model_aware_router_correlations.csv"), index=False)
    eval_summary.to_csv(os.path.join(args.output_dir, "phase4_model_aware_router_eval_summary.csv"), index=False)
    pd.concat(inners, ignore_index=True).to_csv(
        os.path.join(args.output_dir, "phase4_model_aware_router_inner_scores.csv"), index=False
    )
    write_summary(
        os.path.join(args.output_dir, "phase4_model_aware_router_summary.md"),
        {
            "cell_count": len(cells),
            "feature_count": len(feature_cols),
            "router_feature_count": len(router_features),
            "static_regret": float(cells["static_regret_vs_oracle"].mean()),
            "selection_mode": args.selection_mode,
            "evaluation_granularity": args.evaluation_granularity,
            "eval_summary": eval_summary,
            "correlations": corr,
            "verdict": verdict,
        },
    )
    print({"output_dir": args.output_dir, "cells": len(cells), "verdict": verdict.split(":", 1)[0]})


if __name__ == "__main__":
    main()
