# Model-Aware Router Feasibility Summary

- Model-cell rows: 57
- Feature count: 83
- Router feature search count: 20
- Static mean regret vs oracle: 0.0309
- Feature-selection mode: train_score
- Evaluation granularity: config

## Evaluation

| eval_name | cell_count | mean_gain_vs_baseline | mean_gain_vs_static | positive_vs_static_rate | mean_regret_vs_oracle | regret_reduction_vs_static | action_hit_rate | config_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| leave_cell | 57 | 0.08283481403264989 | 0.0023334119141605056 | 0.2982456140350877 | 0.02670129961318756 | 0.13507201712321817 | 0.7192982456140351 | 0.3684210526315789 |
| leave_dataset | 57 | 0.082671801635748 | 0.002245572699238432 | 0.2807017543859649 | 0.02611611452174694 | 0.15402775965566484 | 0.6491228070175439 | 0.2982456140350877 |
| leave_model_scope | 57 | 0.0 | -0.13161969004191373 | 0.12280701754385964 | 0.16905317954516272 | -4.476093961759363 | 0.24561403508771928 | 0.24561403508771928 |
| leave_family | 57 | 0.0 | -0.13161969004191373 | 0.12280701754385964 | 0.16905317954516272 | -4.476093961759363 | 0.24561403508771928 | 0.24561403508771928 |

## Top Correlations

| target | feature | spearman | pearson | abs_spearman |
| --- | --- | --- | --- | --- |
| oracle_gain_vs_baseline | n_heads | -0.7529854282074573 | -0.4391878383061646 | 0.7529854282074573 |
| oracle_gain_vs_baseline | d_model | -0.7485103003355792 | -0.44510549557928414 | 0.7485103003355792 |
| oracle_gain_vs_baseline | train_epochs | -0.7400658416689837 | -0.4088557226256764 | 0.7400658416689837 |
| oracle_gain_vs_baseline | d_ff | -0.7367525403322951 | -0.4451346699066336 | 0.7367525403322951 |
| oracle_gain_vs_baseline | scope_config_count | 0.6126104116285179 | 0.28318592321530833 | 0.6126104116285179 |
| oracle_gain_vs_baseline | d_layers | 0.5735393346764044 | 0.45656049685905215 | 0.5735393346764044 |
| static_regret_vs_oracle | n_heads | -0.5102965908652826 | -0.29500260503713965 | 0.5102965908652826 |
| static_regret_vs_oracle | d_model | -0.49792437454474503 | -0.2961211882557726 | 0.49792437454474503 |
| oracle_gain_vs_baseline | patch_to_seq_ratio | -0.49205188409935324 | -0.273625842369814 | 0.49205188409935324 |
| static_regret_vs_oracle | train_epochs | -0.4858064391532351 | -0.26733657048394854 | 0.4858064391532351 |
| static_regret_vs_oracle | d_ff | -0.4818648047584827 | -0.2925338823522782 | 0.4818648047584827 |
| static_regret_vs_oracle | scope_config_count | 0.4473888202008458 | 0.1925279816662596 | 0.4473888202008458 |
| oracle_gain_vs_baseline | router_freq_seasonal_peak_strength | -0.3833556986691915 | -0.298537491930639 | 0.3833556986691915 |
| oracle_gain_vs_baseline | input_daily_cycle_energy_ratio | -0.3428288716780963 | 0.0459673048285022 | 0.3428288716780963 |
| oracle_gain_vs_baseline | router_multivar_correlation_condition_number | 0.3214061492718445 | -0.1856808585144421 | 0.3214061492718445 |

## Verdict

NO-GO: even with model metadata, the current small cross-architecture matrix is not strong enough for a deployable router claim.
