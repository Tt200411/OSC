# Phase-4 Router Feasibility Summary

- Aggregate rows: 324
- Feature rows: 36
- Input/router feature count: 62
- Router feature search count: 15
- Static baseline by mean rank: `tanh_sin001_all`
- Static mean gain vs GELU: 0.1237
- Oracle mean gain vs GELU: 0.1597
- Static mean regret vs oracle: 0.0472

## Top Correlations

| target | feature | feature_group | n | spearman | pearson | abs_spearman |
| --- | --- | --- | --- | --- | --- | --- |
| gain_tanh_all_outtanh_vs_gelu | router_timefreq_burstiness_index | timefreq | 36 | -0.7783783783783784 | -0.5492749751842525 | 0.7783783783783784 |
| gain_softsign_all_vs_gelu | router_multivar_channel_heterogeneity_cv | multivar | 36 | -0.774774774774775 | -0.7711545816620183 | 0.774774774774775 |
| gain_best_bounded_vs_gelu | router_multivar_channel_heterogeneity_cv | multivar | 36 | -0.7688545688545689 | -0.7970102434948645 | 0.7688545688545689 |
| gain_tanh_sin001_all_vs_gelu | router_multivar_channel_heterogeneity_cv | multivar | 36 | -0.7688545688545689 | -0.7940819981376275 | 0.7688545688545689 |
| static_gain_vs_gelu | router_multivar_channel_heterogeneity_cv | multivar | 36 | -0.7688545688545689 | -0.7940819981376275 | 0.7688545688545689 |
| gain_tanh_all_outtanh_vs_gelu | router_time_late_early_level_shift | time | 36 | -0.7675675675675677 | -0.4169995307007833 | 0.7675675675675677 |
| gain_tanh_all_outtanh_vs_gelu | router_multivar_cross_channel_mean_abs_corr | multivar | 36 | -0.7554697554697556 | -0.5814774900376084 | 0.7554697554697556 |
| gain_tanh_all_vs_gelu | router_multivar_channel_heterogeneity_cv | multivar | 36 | -0.754954954954955 | -0.7998230859267943 | 0.754954954954955 |
| gain_tanh_all_outtanh_vs_gelu | router_time_iqr_mad_ratio | time | 36 | -0.7474903474903476 | -0.27979669395817397 | 0.7474903474903476 |
| gain_best_bounded_vs_gelu | router_multivar_target_channel_dominance | multivar | 36 | 0.7428571428571429 | 0.8093692918889874 | 0.7428571428571429 |
| oracle_gain_vs_gelu | router_multivar_channel_heterogeneity_cv | multivar | 36 | -0.74251882514268 | -0.7669164974444441 | 0.74251882514268 |
| oracle_gain_vs_gelu | router_timefreq_rolling_spectral_entropy_std | timefreq | 36 | -0.7291331503611167 | -0.6123922649558018 | 0.7291331503611167 |
| ... | 8 more rows | | | | | |

## Router Evaluation

| eval_name | cell_count | mean_gain_vs_gelu | median_gain_vs_gelu | mean_gain_vs_static | median_gain_vs_static | positive_vs_static_rate | mean_regret_vs_oracle | regret_reduction_vs_static | top1_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| loco_nested_single_factor | 36 | 0.12627409088770636 | 0.06424710352201934 | 0.00035074323650663306 | 0.0 | 0.2777777777777778 | 0.04429661242444194 | 0.06217249385649978 | 0.19444444444444445 |
| lodo_nested_single_factor | 36 | 0.12475846502470857 | 0.08282084334549453 | 0.0011028673102314545 | 0.0 | 0.25 | 0.0460213476557304 | 0.02565719274908184 | 0.25 |
| family_nested_single_factor | 36 | 0.044357288889201665 | 0.013642556089696467 | -0.1284710513144099 | -0.020413992337746815 | 0.3055555555555556 | 0.18090648290858918 | -2.830068856862283 | 0.08333333333333333 |
| loco_nearest_neighbor | 36 | 0.12034798279283541 | 0.05822820057128877 | -0.006283693219827408 | 0.0 | 0.3888888888888889 | 0.052974298457949306 | -0.12154748395855372 | 0.2777777777777778 |

## Permutation Check

- Observed LOCO mean gain vs static: 0.0004
- Permutation mean: -0.0239
- Permutation p(perm >= observed): 0.0100
- Permutation mode: fixed observed LOCO selected features, shuffled feature-cell alignment.

## Verdict

NO-GO: the input-only factor router does not beat the static tanh-family baseline robustly enough. Keep the paper claim at factor-conditioned diagnostics, not a deployable router.

## Claim Boundary

This audit is input-only and does not use target factors, predictions, errors, or held-out cell metrics as router features. The 972 seed-runs stabilize MSE estimates, but the router sample size is 36 dataset-horizon cells.
