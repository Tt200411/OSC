#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(__file__)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from phase4_model_aware_router import (
    build_model_cells,
    config_action,
    load_scope,
    split_groups,
    standardize_feature_frame,
    usable_features,
)
from phase4_report_utils import markdown_table


CELL_KEYS = ["model_scope", "dataset", "pred_len"]
FREQ_TOKENS = (
    "spectral",
    "freq",
    "frequency",
    "harmonic",
    "daily",
    "cycle",
    "seasonal",
    "period",
    "timefreq",
)
CORE_ACTIONS = {
    "regime_tanh_vs_default",
    "regime_tanh_sin_vs_default",
    "regime_softsign_vs_default",
    "regime_best_bounded_vs_default",
    "placement_encoder_tanh_vs_default",
    "placement_decoder_tanh_vs_default",
    "placement_all_tanh_vs_best_split",
    "placement_output_tanh_vs_best_linear",
    "placement_output_tanh_matched",
}
ACTION_CONFIG_CANDIDATES = {
    "regime_tanh_vs_default": ["tanh_all", "tanh_ffn_linear"],
    "regime_tanh_sin_vs_default": ["tanh_sin001_all", "tanh_sin001_ffn_linear"],
    "regime_softsign_vs_default": ["softsign_all", "softsign_ffn_linear"],
    "regime_swish_vs_default": ["swish_all", "swish_ffn_linear"],
    "regime_relu_vs_default": ["relu_all", "relu_ffn_linear"],
    "regime_tanh_sin_vs_tanh": ["tanh_sin001_all", "tanh_sin001_ffn_linear"],
    "regime_softsign_vs_tanh": ["softsign_all", "softsign_ffn_linear"],
    "regime_swish_vs_gelu": ["swish_all", "swish_ffn_linear"],
    "regime_relu_vs_gelu": ["relu_all", "relu_ffn_linear"],
    "placement_encoder_tanh_vs_default": ["enc_tanh_dec_gelu"],
    "placement_decoder_tanh_vs_default": ["enc_gelu_dec_tanh"],
    "placement_all_tanh_vs_best_split": ["tanh_all"],
}
BOUNDED_CONFIGS = [
    "tanh_sin001_all",
    "tanh_all",
    "softsign_all",
    "tanh_sin001_ffn_linear",
    "tanh_ffn_linear",
    "softsign_ffn_linear",
]


def has_config(group, name):
    return name in set(group["config_name"].astype(str))


def get_mse(group, config):
    rows = group[group["config_name"].astype(str) == str(config)]
    if rows.empty:
        return None
    return float(rows.iloc[0]["mse_mean"])


def best_config(group, mask):
    rows = group[mask(group)].copy()
    if rows.empty:
        return None, None
    row = rows.sort_values("mse_mean").iloc[0]
    return str(row["config_name"]), float(row["mse_mean"])


def matched_linear(out_config):
    name = str(out_config)
    if name == "tanh_all_outtanh":
        return "tanh_all"
    if name.endswith("_outtanh"):
        return name.replace("_outtanh", "_linear")
    return None


def add_pair(rows, group, cell_meta, action_name, action_family, action_config, comparator_config):
    action_mse = get_mse(group, action_config)
    comparator_mse = get_mse(group, comparator_config)
    if action_mse is None or comparator_mse is None or comparator_mse <= 0:
        return
    rows.append(
        {
            **cell_meta,
            "action_name": action_name,
            "action_family": action_family,
            "action_config": str(action_config),
            "comparator_config": str(comparator_config),
            "action_mse": action_mse,
            "comparator_mse": comparator_mse,
            "uplift": (comparator_mse - action_mse) / comparator_mse,
        }
    )


def make_pair_rows(results):
    rows = []
    for key, group in results.groupby(CELL_KEYS, dropna=False):
        group = group.copy()
        cell_meta = dict(zip(CELL_KEYS, key))
        for col in ["model_family", "search_space", "baseline_config"]:
            cell_meta[col] = group[col].iloc[0]
        baseline = str(group["baseline_config"].iloc[0])
        names = set(group["config_name"].astype(str))

        def name_mask(token, exclude_out=True):
            def _mask(df):
                s = df["config_name"].astype(str)
                m = s.str.contains(token, regex=False)
                if exclude_out:
                    m &= ~s.str.contains("outtanh", regex=False)
                return m

            return _mask

        # Regime choices relative to the model's default GELU/baseline config.
        for action_name, candidates in [
            ("regime_tanh_vs_default", ["tanh_all", "tanh_ffn_linear"]),
            ("regime_tanh_sin_vs_default", ["tanh_sin001_all", "tanh_sin001_ffn_linear"]),
            ("regime_softsign_vs_default", ["softsign_all", "softsign_ffn_linear"]),
            ("regime_swish_vs_default", ["swish_all", "swish_ffn_linear"]),
            ("regime_relu_vs_default", ["relu_all", "relu_ffn_linear"]),
        ]:
            for candidate in candidates:
                if candidate in names:
                    add_pair(rows, group, cell_meta, action_name, "regime", candidate, baseline)
                    break

        bounded_config, _ = best_config(
            group,
            lambda df: df["action"].eq("bounded_linear") & ~df["config_name"].astype(str).str.contains("outtanh"),
        )
        if bounded_config is not None:
            add_pair(rows, group, cell_meta, "regime_best_bounded_vs_default", "regime", bounded_config, baseline)

        other_config, _ = best_config(group, lambda df: df["action"].eq("other_linear"))
        if other_config is not None:
            add_pair(rows, group, cell_meta, "regime_best_unbounded_vs_default", "regime", other_config, baseline)

        # Within-family choices: useful for deciding regime variants.
        for action_name, action_cfgs, comp_cfgs in [
            ("regime_tanh_sin_vs_tanh", ["tanh_sin001_all", "tanh_sin001_ffn_linear"], ["tanh_all", "tanh_ffn_linear"]),
            ("regime_softsign_vs_tanh", ["softsign_all", "softsign_ffn_linear"], ["tanh_all", "tanh_ffn_linear"]),
            ("regime_swish_vs_gelu", ["swish_all", "swish_ffn_linear"], ["gelu_all", "gelu_ffn_linear"]),
            ("regime_relu_vs_gelu", ["relu_all", "relu_ffn_linear"], ["gelu_all", "gelu_ffn_linear"]),
        ]:
            for action_cfg, comp_cfg in zip(action_cfgs, comp_cfgs):
                if action_cfg in names and comp_cfg in names:
                    add_pair(rows, group, cell_meta, action_name, "regime", action_cfg, comp_cfg)
                    break

        # Placement: output tanh, matched to same base activation where possible.
        matched_rows = []
        for out_config in sorted(n for n in names if "outtanh" in n):
            linear = matched_linear(out_config)
            if linear and linear in names:
                out_mse = get_mse(group, out_config)
                lin_mse = get_mse(group, linear)
                if out_mse is not None and lin_mse is not None and lin_mse > 0:
                    matched_rows.append(
                        {
                            "action_config": out_config,
                            "comparator_config": linear,
                            "uplift": (lin_mse - out_mse) / lin_mse,
                        }
                    )
        if matched_rows:
            best_match = sorted(matched_rows, key=lambda r: r["uplift"], reverse=True)[0]
            add_pair(
                rows,
                group,
                cell_meta,
                "placement_output_tanh_matched",
                "placement",
                best_match["action_config"],
                best_match["comparator_config"],
            )

        out_config, _ = best_config(group, lambda df: df["action"].eq("output_tanh"))
        lin_config, _ = best_config(group, lambda df: ~df["config_name"].astype(str).str.contains("outtanh"))
        if out_config is not None and lin_config is not None:
            add_pair(rows, group, cell_meta, "placement_output_tanh_vs_best_linear", "placement", out_config, lin_config)

        # Informer-specific encoder/decoder placement probes.
        if has_config(group, "enc_tanh_dec_gelu"):
            add_pair(rows, group, cell_meta, "placement_encoder_tanh_vs_default", "placement", "enc_tanh_dec_gelu", baseline)
        if has_config(group, "enc_gelu_dec_tanh"):
            add_pair(rows, group, cell_meta, "placement_decoder_tanh_vs_default", "placement", "enc_gelu_dec_tanh", baseline)
        if has_config(group, "tanh_all") and has_config(group, "enc_tanh_dec_gelu") and has_config(group, "enc_gelu_dec_tanh"):
            enc_mse = get_mse(group, "enc_tanh_dec_gelu")
            dec_mse = get_mse(group, "enc_gelu_dec_tanh")
            comparator = "enc_tanh_dec_gelu" if enc_mse <= dec_mse else "enc_gelu_dec_tanh"
            add_pair(rows, group, cell_meta, "placement_all_tanh_vs_best_split", "placement", "tanh_all", comparator)

    return pd.DataFrame(rows)


def load_result_scopes(args):
    scopes = [
        load_scope(args.informer, "Informer-fullgrid", "Informer", "gelu_all", "fullgrid"),
        load_scope(args.patchtst, "PatchTST-transfer", "PatchTST", "gelu_ffn_linear", "transfer"),
        load_scope(args.itransformer_stage1, "iTransformer-stage1", "iTransformer", "gelu_ffn_linear", "ffn_only"),
        load_scope(args.itransformer_placeprobe, "iTransformer-placeprobe", "iTransformer", "gelu_ffn_linear", "placement_probe"),
    ]
    results = pd.concat(scopes, ignore_index=True)
    if args.scope_filter:
        keep = set(args.scope_filter)
        results = results[results["model_scope"].isin(keep)].copy()
    results["action"] = results.apply(lambda row: config_action(row["config_name"], row["baseline_config"]), axis=1)
    return results


def load_results(args):
    results = load_result_scopes(args)
    cells, _ = build_model_cells(results)
    cells = standardize_feature_frame(cells, args.factors_dir, args.lee_root)
    pairs = make_pair_rows(results)
    pairs = pairs.merge(cells, on=CELL_KEYS + ["model_family", "search_space", "baseline_config"], how="left")
    return pairs


def feature_groups(pairs, top_k, include_topology=False):
    all_features = usable_features(pairs)
    data = [c for c in all_features if c.startswith(("input_", "router_"))]
    freq = [c for c in data if any(token in c for token in FREQ_TOKENS)]
    stats = [c for c in data if c not in set(freq)]
    topology = [c for c in all_features if not c.startswith(("input_", "router_"))]
    groups = {
        "data": data[:top_k],
        "frequency": freq[:top_k],
        "statistical": stats[:top_k],
        "all_input": data[:top_k],
    }
    if include_topology:
        groups["topology"] = topology[:top_k]
        groups["all_with_topology"] = all_features[:top_k]
    return groups


def split_pair_groups(df, mode):
    reset = df.reset_index(drop=True)
    if mode == "leave_cell":
        groups = []
        for row in reset[CELL_KEYS].drop_duplicates().itertuples(index=False):
            mask = (
                (reset["model_scope"] == row.model_scope)
                & (reset["dataset"] == row.dataset)
                & (reset["pred_len"] == row.pred_len)
            )
            groups.append((f"{row.model_scope}:{row.dataset}/{int(row.pred_len)}", reset.index[mask].tolist()))
        return groups
    if mode == "leave_pair_cell":
        groups = []
        for row in reset[CELL_KEYS + ["action_name"]].drop_duplicates().itertuples(index=False):
            mask = (
                (reset["model_scope"] == row.model_scope)
                & (reset["dataset"] == row.dataset)
                & (reset["pred_len"] == row.pred_len)
                & (reset["action_name"] == row.action_name)
            )
            groups.append((f"{row.action_name}:{row.model_scope}:{row.dataset}/{int(row.pred_len)}", reset.index[mask].tolist()))
        return groups
    if mode == "leave_dataset":
        return [(d, reset.index[reset["dataset"] == d].tolist()) for d in sorted(reset["dataset"].unique())]
    if mode == "leave_horizon":
        return [(str(h), reset.index[reset["pred_len"] == h].tolist()) for h in sorted(reset["pred_len"].unique())]
    if mode == "leave_family":
        return [(m, reset.index[reset["model_family"] == m].tolist()) for m in sorted(reset["model_family"].unique())]
    if mode == "leave_model_scope":
        return [(s, reset.index[reset["model_scope"] == s].tolist()) for s in sorted(reset["model_scope"].unique())]
    raise ValueError(mode)


def candidate_thresholds(values, max_thresholds=12):
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.nunique() < 3:
        return []
    qs = np.linspace(0.1, 0.9, max_thresholds)
    return sorted(set(float(x) for x in np.nanquantile(clean.to_numpy(float), qs)))


def condition(values, direction, threshold):
    vals = pd.to_numeric(values, errors="coerce")
    if direction == "high":
        return vals >= threshold
    return vals <= threshold


def score_rule(df, fire, margin):
    if len(df) == 0:
        return None
    fire = np.asarray(fire, dtype=bool)
    fired = df[fire]
    policy_uplift = float(np.where(fire, df["uplift"].to_numpy(float), 0.0).mean())
    if fired.empty:
        return {
            "fired_count": 0,
            "fired_rate": 0.0,
            "precision": np.nan,
            "mean_uplift_if_fired": np.nan,
            "policy_mean_uplift": 0.0,
        }
    return {
        "fired_count": int(fire.sum()),
        "fired_rate": float(fire.mean()),
        "precision": float((fired["uplift"] > margin).mean()),
        "mean_uplift_if_fired": float(fired["uplift"].mean()),
        "policy_mean_uplift": policy_uplift,
    }


def fit_action_rule(train, features, margin, min_support, min_precision):
    best = None
    for feature in features:
        thresholds = candidate_thresholds(train[feature])
        for threshold in thresholds:
            for direction in ["high", "low"]:
                fire = condition(train[feature], direction, threshold)
                if int(fire.sum()) < min_support:
                    continue
                score = score_rule(train, fire, margin)
                if score is None:
                    continue
                if score["mean_uplift_if_fired"] <= 0:
                    continue
                if score["precision"] < min_precision:
                    continue
                objective = score["policy_mean_uplift"]
                candidate = {
                    "feature": feature,
                    "direction": direction,
                    "threshold": float(threshold),
                    "objective": float(objective),
                    **score,
                }
                if best is None or (candidate["objective"], candidate["precision"], candidate["fired_rate"]) > (
                    best["objective"],
                    best["precision"],
                    best["fired_rate"],
                ):
                    best = candidate
    if best is None:
        return {
            "feature": "abstain",
            "direction": "none",
            "threshold": np.nan,
            "objective": 0.0,
            "fired_count": 0,
            "fired_rate": 0.0,
            "precision": np.nan,
            "mean_uplift_if_fired": np.nan,
            "policy_mean_uplift": 0.0,
        }
    return best


def apply_rule(df, rule):
    if rule["feature"] == "abstain" or df.empty:
        return np.zeros(len(df), dtype=bool)
    return condition(df[rule["feature"]], rule["direction"], rule["threshold"]).to_numpy(bool)


def evaluate_router(pairs, feature_sets, mode, margin, min_support, min_precision, min_pairs):
    all_predictions = []
    rule_rows = []
    for feature_set, features in feature_sets.items():
        if not features:
            continue
        for action_name, action_df in pairs.groupby("action_name", dropna=False):
            if len(action_df) < min_pairs:
                continue
            reset = action_df.reset_index(drop=True)
            for fold, test_idx in split_pair_groups(reset, mode):
                test = reset.loc[test_idx]
                train = reset.drop(index=test_idx)
                if len(train) < min_pairs // 2 or test.empty:
                    continue
                rule = fit_action_rule(train, features, margin, min_support, min_precision)
                fire = apply_rule(test, rule)
                pred = test[CELL_KEYS + ["model_family", "action_name", "action_family", "action_config", "comparator_config", "uplift"]].copy()
                pred["feature_set"] = feature_set
                pred["eval_name"] = mode
                pred["fold"] = str(fold)
                pred["recommend"] = fire
                pred["policy_uplift"] = np.where(fire, pred["uplift"], 0.0)
                pred["selected_feature"] = rule["feature"]
                pred["rule_direction"] = rule["direction"]
                pred["rule_threshold"] = rule["threshold"]
                pred["rule_objective"] = rule["objective"]
                pred["rule_precision"] = rule["precision"]
                pred["rule_mean_uplift_if_fired"] = rule["mean_uplift_if_fired"]
                all_predictions.append(pred)
                rule_rows.append(
                    {
                        "feature_set": feature_set,
                        "eval_name": mode,
                        "fold": str(fold),
                        "action_name": action_name,
                        "action_family": str(test["action_family"].iloc[0]),
                        **rule,
                    }
                )
    pred_all = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    rules = pd.DataFrame(rule_rows)
    return pred_all, rules


def summarize_predictions(pred, margin):
    rows = []
    if pred.empty:
        return pd.DataFrame()
    for key, group in pred.groupby(["eval_name", "feature_set", "action_family", "action_name"], dropna=False):
        eval_name, feature_set, action_family, action_name = key
        fired = group[group["recommend"]]
        positive_total = int((group["uplift"] > margin).sum())
        rows.append(
            {
                "eval_name": eval_name,
                "feature_set": feature_set,
                "action_family": action_family,
                "action_name": action_name,
                "pair_count": int(len(group)),
                "always_mean_uplift": float(group["uplift"].mean()),
                "always_positive_rate": float((group["uplift"] > margin).mean()),
                "policy_mean_uplift": float(group["policy_uplift"].mean()),
                "policy_excess_vs_always": float(group["policy_uplift"].mean() - group["uplift"].mean()),
                "recommend_rate": float(group["recommend"].mean()),
                "precision_if_recommended": float((fired["uplift"] > margin).mean()) if not fired.empty else np.nan,
                "mean_uplift_if_recommended": float(fired["uplift"].mean()) if not fired.empty else np.nan,
                "positive_recall": float(((group["recommend"]) & (group["uplift"] > margin)).sum() / positive_total)
                if positive_total
                else np.nan,
                "top_selected_features": ";".join(group["selected_feature"].value_counts().head(5).index.astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["eval_name", "action_family", "policy_mean_uplift", "precision_if_recommended"],
        ascending=[True, True, False, False],
    )


def cell_id_frame(df):
    return df[CELL_KEYS].drop_duplicates().copy()


def same_cells_mask(df, cells):
    if cells.empty:
        return pd.Series(False, index=df.index)
    marker = cells.assign(__cell_marker=True)
    merged = df[CELL_KEYS].merge(marker, on=CELL_KEYS, how="left")
    return merged["__cell_marker"].eq(True).to_numpy(bool)


def rank_static_configs(results, train_cells):
    train = results[same_cells_mask(results, train_cells)].copy()
    if train.empty:
        return {}
    train["rank_in_cell"] = train.groupby(CELL_KEYS, dropna=False)["mse_mean"].rank(method="min")
    ranked = (
        train.groupby(["model_scope", "config_name"], dropna=False)["rank_in_cell"]
        .mean()
        .reset_index()
        .sort_values(["model_scope", "rank_in_cell", "config_name"])
    )
    out = {}
    for scope, group in ranked.groupby("model_scope", dropna=False):
        out[str(scope)] = str(group.iloc[0]["config_name"])
    return out


def best_train_config(results, train_cells, scope, candidates):
    train = results[(results["model_scope"].astype(str) == str(scope)) & same_cells_mask(results, train_cells)].copy()
    train = train[train["config_name"].astype(str).isin(candidates)].copy()
    if train.empty:
        return None
    train["rank_in_cell"] = train.groupby(CELL_KEYS, dropna=False)["mse_mean"].rank(method="min")
    ranked = (
        train.groupby("config_name", dropna=False)["rank_in_cell"]
        .mean()
        .reset_index()
        .sort_values(["rank_in_cell", "config_name"])
    )
    return str(ranked.iloc[0]["config_name"])


def action_to_exact_config(action_name, available_configs, results, train_cells, scope):
    available = set(str(x) for x in available_configs)
    if action_name == "regime_best_bounded_vs_default":
        chosen = best_train_config(results, train_cells, scope, BOUNDED_CONFIGS)
        if chosen in available:
            return chosen
    if action_name == "regime_best_unbounded_vs_default":
        chosen = best_train_config(results, train_cells, scope, ["swish_all", "relu_all", "swish_ffn_linear", "relu_ffn_linear"])
        if chosen in available:
            return chosen
    if action_name in {"placement_output_tanh_vs_best_linear", "placement_output_tanh_matched"}:
        outtanh = sorted(c for c in available if "outtanh" in c)
        chosen = best_train_config(results, train_cells, scope, outtanh)
        if chosen in available:
            return chosen
        return outtanh[0] if outtanh else None
    for candidate in ACTION_CONFIG_CANDIDATES.get(action_name, []):
        if candidate in available:
            return candidate
    return None


def build_cell_metrics(results):
    rows = []
    for key, group in results.groupby(CELL_KEYS, dropna=False):
        meta = dict(zip(CELL_KEYS, key))
        group = group.copy()
        baseline_config = str(group["baseline_config"].iloc[0])
        baseline_rows = group[group["config_name"].astype(str) == baseline_config]
        if baseline_rows.empty:
            continue
        oracle = group.sort_values("mse_mean").iloc[0]
        rows.append(
            {
                **meta,
                "model_family": str(group["model_family"].iloc[0]),
                "search_space": str(group["search_space"].iloc[0]),
                "baseline_config": baseline_config,
                "baseline_mse": float(baseline_rows.iloc[0]["mse_mean"]),
                "oracle_config": str(oracle["config_name"]),
                "oracle_action": str(oracle["action"]),
                "oracle_mse": float(oracle["mse_mean"]),
                "available_configs": tuple(sorted(group["config_name"].astype(str).unique())),
            }
        )
    return pd.DataFrame(rows)


def mse_for_config(results, cell, config):
    rows = results[
        (results["model_scope"].astype(str) == str(cell["model_scope"]))
        & (results["dataset"].astype(str) == str(cell["dataset"]))
        & (results["pred_len"].astype(int) == int(cell["pred_len"]))
        & (results["config_name"].astype(str) == str(config))
    ]
    if rows.empty:
        return None
    return float(rows.iloc[0]["mse_mean"])


def score_cell_prediction(row, margin):
    baseline_gain = (row["baseline_mse"] - row["chosen_mse"]) / row["baseline_mse"]
    static_gain = (row["static_mse"] - row["chosen_mse"]) / row["static_mse"] if row["static_mse"] > 0 else np.nan
    oracle_gain = (row["baseline_mse"] - row["oracle_mse"]) / row["baseline_mse"] if row["baseline_mse"] > 0 else np.nan
    regret = (row["chosen_mse"] - row["oracle_mse"]) / row["oracle_mse"] if row["oracle_mse"] > 0 else np.nan
    capture = baseline_gain / oracle_gain if oracle_gain and oracle_gain > 1e-12 else np.nan
    return {
        "gain_vs_baseline": baseline_gain,
        "gain_vs_static": static_gain,
        "regret_vs_oracle": regret,
        "oracle_capture_ratio": capture,
        "positive_vs_baseline": baseline_gain > margin,
        "positive_vs_static": static_gain > margin if not pd.isna(static_gain) else False,
    }


def combine_pair_predictions_to_cells(pred, pairs, results, fallback_policies, margin, core_only):
    if pred.empty:
        return pd.DataFrame(), pd.DataFrame()
    if core_only:
        pred = pred[pred["action_name"].isin(CORE_ACTIONS)].copy()
    cells = build_cell_metrics(results)
    all_cells = cell_id_frame(cells)
    rows = []
    for (eval_name, feature_set, fold), group in pred.groupby(["eval_name", "feature_set", "fold"], dropna=False):
        test_cells = cell_id_frame(group)
        train_cells = all_cells[~same_cells_mask(all_cells, test_cells)].copy()
        static_by_scope = rank_static_configs(results, train_cells)
        for fallback_policy in fallback_policies:
            for _, cell in cells[same_cells_mask(cells, test_cells)].iterrows():
                cell_group = group[
                    (group["model_scope"].astype(str) == str(cell["model_scope"]))
                    & (group["dataset"].astype(str) == str(cell["dataset"]))
                    & (group["pred_len"].astype(int) == int(cell["pred_len"]))
                ].copy()
                recommendations = cell_group[cell_group["recommend"]].copy()
                candidates = []
                for _, rec in recommendations.iterrows():
                    config = action_to_exact_config(
                        str(rec["action_name"]),
                        cell["available_configs"],
                        results,
                        train_cells,
                        cell["model_scope"],
                    )
                    if not config:
                        continue
                    chosen_mse = mse_for_config(results, cell, config)
                    if chosen_mse is None:
                        continue
                    candidates.append(
                        {
                            "chosen_config": config,
                            "chosen_mse": chosen_mse,
                            "chosen_action_name": str(rec["action_name"]),
                            "chosen_action_family": str(rec["action_family"]),
                            "rule_objective": float(rec.get("rule_objective", 0.0)),
                            "rule_precision": float(rec.get("rule_precision", np.nan)),
                        }
                    )
                abstained = not candidates
                if candidates:
                    chosen = sorted(
                        candidates,
                        key=lambda item: (
                            item["rule_objective"],
                            -1.0 if pd.isna(item["rule_precision"]) else item["rule_precision"],
                            -item["chosen_mse"],
                        ),
                        reverse=True,
                    )[0]
                else:
                    if fallback_policy == "train_static":
                        fallback_config = static_by_scope.get(str(cell["model_scope"]), cell["baseline_config"])
                        if fallback_config not in set(cell["available_configs"]):
                            fallback_config = cell["baseline_config"]
                    else:
                        fallback_config = cell["baseline_config"]
                    chosen = {
                        "chosen_config": fallback_config,
                        "chosen_mse": mse_for_config(results, cell, fallback_config),
                        "chosen_action_name": "abstain",
                        "chosen_action_family": "abstain",
                        "rule_objective": 0.0,
                        "rule_precision": np.nan,
                    }
                static_config = static_by_scope.get(str(cell["model_scope"]), cell["baseline_config"])
                if static_config not in set(cell["available_configs"]):
                    static_config = cell["baseline_config"]
                static_mse = mse_for_config(results, cell, static_config)
                out = {
                    "eval_name": eval_name,
                    "feature_set": feature_set,
                    "fold": fold,
                    "fallback_policy": fallback_policy,
                    "model_scope": cell["model_scope"],
                    "model_family": cell["model_family"],
                    "dataset": cell["dataset"],
                    "pred_len": cell["pred_len"],
                    "baseline_config": cell["baseline_config"],
                    "baseline_mse": cell["baseline_mse"],
                    "static_config": static_config,
                    "static_mse": static_mse,
                    "oracle_config": cell["oracle_config"],
                    "oracle_action": cell["oracle_action"],
                    "oracle_mse": cell["oracle_mse"],
                    "chosen_config": chosen["chosen_config"],
                    "chosen_mse": chosen["chosen_mse"],
                    "chosen_action_name": chosen["chosen_action_name"],
                    "chosen_action_family": chosen["chosen_action_family"],
                    "rule_objective": chosen["rule_objective"],
                    "rule_precision": chosen["rule_precision"],
                    "abstained": abstained,
                    "recommendation_count": len(candidates),
                }
                out.update(score_cell_prediction(out, margin))
                out["config_hit"] = out["chosen_config"] == out["oracle_config"]
                out["action_hit"] = config_action(out["chosen_config"], out["baseline_config"]) == out["oracle_action"]
                rows.append(out)
    cell_predictions = pd.DataFrame(rows)
    summary = summarize_cell_predictions(cell_predictions)
    return cell_predictions, summary


def summarize_cell_predictions(cell_predictions):
    if cell_predictions.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["eval_name", "feature_set", "fallback_policy"]
    for key, group in cell_predictions.groupby(group_cols, dropna=False):
        eval_name, feature_set, fallback_policy = key
        chosen = group[~group["abstained"]]
        placement = chosen[chosen["chosen_action_family"] == "placement"]
        rows.append(
            {
                "eval_name": eval_name,
                "feature_set": feature_set,
                "fallback_policy": fallback_policy,
                "cell_count": int(len(group)),
                "mean_mse": float(group["chosen_mse"].mean()),
                "mean_gain_vs_baseline": float(group["gain_vs_baseline"].mean()),
                "mean_gain_vs_static": float(group["gain_vs_static"].mean()),
                "positive_vs_baseline_rate": float(group["positive_vs_baseline"].mean()),
                "positive_vs_static_rate": float(group["positive_vs_static"].mean()),
                "mean_regret_vs_oracle": float(group["regret_vs_oracle"].mean()),
                "mean_oracle_capture_ratio": float(group["oracle_capture_ratio"].replace([np.inf, -np.inf], np.nan).mean()),
                "config_hit_rate": float(group["config_hit"].mean()),
                "action_hit_rate": float(group["action_hit"].mean()),
                "abstain_rate": float(group["abstained"].mean()),
                "mean_recommendation_count": float(group["recommendation_count"].mean()),
                "placement_choice_rate": float((group["chosen_action_family"] == "placement").mean()),
                "placement_precision_vs_static": float(placement["positive_vs_static"].mean()) if not placement.empty else np.nan,
                "top_chosen_configs": ";".join(group["chosen_config"].value_counts().head(5).index.astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["eval_name", "fallback_policy", "mean_gain_vs_static", "mean_regret_vs_oracle"],
        ascending=[True, True, False, True],
    )


def write_cell_summary(path, summary, cell_predictions):
    best = summary.sort_values(["eval_name", "mean_gain_vs_static", "mean_regret_vs_oracle"], ascending=[True, False, True])
    action_summary = (
        cell_predictions.groupby(["eval_name", "feature_set", "fallback_policy", "chosen_action_family"], dropna=False)
        .agg(
            count=("chosen_action_family", "size"),
            mean_gain_vs_baseline=("gain_vs_baseline", "mean"),
            mean_gain_vs_static=("gain_vs_static", "mean"),
            mean_regret_vs_oracle=("regret_vs_oracle", "mean"),
            precision_vs_static=("positive_vs_static", "mean"),
        )
        .reset_index()
        .sort_values(["eval_name", "mean_gain_vs_static"], ascending=[True, False])
        if not cell_predictions.empty
        else pd.DataFrame()
    )
    lines = [
        "# Phase-5 Router Cell-Level Tables",
        "",
        "## Table 2 Candidate: Router vs Baseline / Static / Oracle",
        "",
        markdown_table(best),
        "",
        "## Table 3 Candidate: Action and Placement Precision",
        "",
        markdown_table(action_summary),
        "",
        "## Table 4 Candidate: Split Stability",
        "",
        markdown_table(
            summary.groupby(["eval_name", "fallback_policy"], dropna=False)
            .agg(
                feature_sets=("feature_set", "nunique"),
                best_gain_vs_static=("mean_gain_vs_static", "max"),
                best_regret_vs_oracle=("mean_regret_vs_oracle", "min"),
                best_capture_ratio=("mean_oracle_capture_ratio", "max"),
                min_abstain_rate=("abstain_rate", "min"),
            )
            .reset_index()
        ),
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_summary(path, summary, pair_counts, rules, margin):
    filtered = summary[
        (summary["policy_mean_uplift"] > 0)
        & (summary["recommend_rate"] > 0)
        & (summary["precision_if_recommended"].fillna(0) >= 0.55)
    ].copy()
    filtered = filtered.sort_values(["eval_name", "policy_mean_uplift"], ascending=[True, False]).head(30)
    rule_counts = (
        rules[rules["feature"] != "abstain"]
        .groupby(["action_name", "feature"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(30)
        if not rules.empty
        else pd.DataFrame()
    )
    lines = [
        "# Pairwise Factor Router Summary",
        "",
        f"- Positive margin: {margin}",
        "",
        "## Pair Availability",
        "",
        markdown_table(pair_counts),
        "",
        "## Best Router Signals",
        "",
        markdown_table(filtered),
        "",
        "## Frequently Selected Factor Rules",
        "",
        markdown_table(rule_counts),
        "",
        "## Full Summary",
        "",
        markdown_table(summary),
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    results = load_result_scopes(args)
    cells, _ = build_model_cells(results)
    cells = standardize_feature_frame(cells, args.factors_dir, args.lee_root)
    pairs = make_pair_rows(results)
    pairs = pairs.merge(cells, on=CELL_KEYS + ["model_family", "search_space", "baseline_config"], how="left")
    pairs["positive"] = pairs["uplift"] > args.margin
    pair_counts = (
        pairs.groupby(["action_family", "action_name"], dropna=False)
        .agg(pair_count=("uplift", "size"), mean_uplift=("uplift", "mean"), positive_rate=("positive", "mean"))
        .reset_index()
        .sort_values(["action_family", "pair_count", "mean_uplift"], ascending=[True, False, False])
    )
    feature_sets = feature_groups(pairs, args.top_k, include_topology=args.include_topology)
    predictions = []
    rules = []
    for mode in args.modes:
        pred, rule = evaluate_router(
            pairs,
            feature_sets,
            mode,
            args.margin,
            args.min_support,
            args.min_precision,
            args.min_pairs,
        )
        if not pred.empty:
            predictions.append(pred)
        if not rule.empty:
            rules.append(rule)
    pred_all = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    rules_all = pd.concat(rules, ignore_index=True) if rules else pd.DataFrame()
    summary = summarize_predictions(pred_all, args.margin)
    fallback_policies = [item.strip() for item in args.fallback_policies.split(",") if item.strip()]
    cell_predictions, cell_summary = combine_pair_predictions_to_cells(
        pred_all,
        pairs,
        results,
        fallback_policies=fallback_policies,
        margin=args.margin,
        core_only=args.core_only,
    )

    pairs.to_csv(os.path.join(args.output_dir, "phase4_pairwise_factor_router_pairs.csv"), index=False)
    pair_counts.to_csv(os.path.join(args.output_dir, "phase4_pairwise_factor_router_pair_counts.csv"), index=False)
    pred_all.to_csv(os.path.join(args.output_dir, "phase4_pairwise_factor_router_predictions.csv"), index=False)
    rules_all.to_csv(os.path.join(args.output_dir, "phase4_pairwise_factor_router_rules.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, "phase4_pairwise_factor_router_summary.csv"), index=False)
    cell_predictions.to_csv(os.path.join(args.output_dir, "phase5_router_cell_predictions.csv"), index=False)
    cell_summary.to_csv(os.path.join(args.output_dir, "phase5_router_cell_summary.csv"), index=False)
    write_summary(
        os.path.join(args.output_dir, "phase4_pairwise_factor_router_summary.md"),
        summary,
        pair_counts,
        rules_all,
        args.margin,
    )
    write_cell_summary(
        os.path.join(args.output_dir, "phase5_router_cell_tables.md"),
        cell_summary,
        cell_predictions,
    )
    print(
        {
            "output_dir": args.output_dir,
            "pairs": len(pairs),
            "summary_rows": len(summary),
            "cell_prediction_rows": len(cell_predictions),
            "cell_summary_rows": len(cell_summary),
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Pairwise factor-uplift router for activation regime and placement")
    parser.add_argument("--factors_dir", required=True)
    parser.add_argument("--output_dir", default="phase4_remote_results/analysis_current/factors_router/pairwise_uplift")
    parser.add_argument("--lee_root", default="lee_ocil")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--margin", type=float, default=0.002)
    parser.add_argument("--min_support", type=int, default=4)
    parser.add_argument("--min_precision", type=float, default=0.55)
    parser.add_argument("--min_pairs", type=int, default=10)
    parser.add_argument("--modes", nargs="+", default=["leave_cell", "leave_dataset", "leave_horizon", "leave_family"])
    parser.add_argument("--fallback_policies", default="gelu,train_static")
    parser.add_argument("--core_only", action="store_true", help="Only combine the Phase-5 core router actions in cell-level tables")
    parser.add_argument("--include_topology", action="store_true", help="Include topology/model metadata feature sets in addition to input factors")
    parser.add_argument("--scope_filter", nargs="+", default=[], help="Optional model_scope filter, e.g. Informer-fullgrid")
    parser.add_argument("--informer", default="phase4_remote_results/analysis_current/phase4_protocol_aggregate.csv")
    parser.add_argument("--patchtst", default="patchtst_remote_results/analysis_patchtst_transfer_actfix/patchtst_transfer_aggregate.csv")
    parser.add_argument("--itransformer_stage1", default="itransformer_remote_results/analysis_itransformer_stage1_actfix3/itransformer_protocol_aggregate.csv")
    parser.add_argument("--itransformer_placeprobe", default="itransformer_remote_results/analysis_itransformer_placeprobe_20260522/itransformer_protocol_aggregate.csv")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
