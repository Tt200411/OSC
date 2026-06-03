# Phase-5 Factor Signal Audit

Scope: Weather, Exchange, and ILI 3-seed oracle matrix. The audit checks whether input/statistical/time-frequency factors are aligned with activation regime and placement uplift.

## Verdict

There is interpretable signal, but it is not yet strong enough for a deployable exact zero-shot router. The strongest reproducible pattern is placement-level: decoder-side tanh is broadly safe/positive except one Weather-96 failure, while static all-layer bounded regimes and output tanh need abstain gates. Leave-dataset threshold rules only remain positive for decoder-side placement and oracle gain, so the current factors are better as calibration/abstain features than as a standalone exact regime+placement selector.

## Action Uplift Summary

| action | target | n | mean_uplift | median_uplift | positive_rate | min | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_gain_vs_gelu | oracle_gain_vs_gelu | 9 | 0.024035504463226345 | 0.02059202609431526 | 1.0 | 0.0026294205080345523 | 0.04870810214188847 |
| placement_all_tanh_vs_best_split | gain_alltanh_vs_best_split | 9 | -0.12849767731878436 | -0.08903747388002622 | 0.3333333333333333 | -0.45359169864335763 | 0.12548543347989194 |
| placement_decoder_tanh_vs_gelu | gain_enc_gelu_dec_tanh_vs_gelu | 9 | -0.004085260911053611 | 0.007583739654931813 | 0.8888888888888888 | -0.12477278407830826 | 0.03456457725671016 |
| placement_encoder_tanh_vs_gelu | gain_enc_tanh_dec_gelu_vs_gelu | 9 | -0.12351031609224163 | -0.10853791478604927 | 0.3333333333333333 | -0.32080647515926863 | 0.04013335431990009 |
| placement_output_tanh_vs_tanh | gain_outtanh_vs_tanh | 9 | -0.04678131897505056 | -0.030974965351492023 | 0.3333333333333333 | -0.23788630310494926 | 0.030774644473547278 |
| regime_softsign_vs_gelu | gain_softsign_all_vs_gelu | 9 | -0.12714663915101776 | -0.06063594407067785 | 0.3333333333333333 | -0.4948052687398383 | 0.028722801484444956 |
| regime_tanh_sin_vs_gelu | gain_tanh_sin001_all_vs_gelu | 9 | -0.11119379614786123 | -0.05112132166548884 | 0.4444444444444444 | -0.42203526591931256 | 0.04870810214188847 |
| regime_tanh_vs_gelu | gain_tanh_all_vs_gelu | 9 | -0.11901814782161968 | -0.051395353978647584 | 0.4444444444444444 | -0.43382713961213815 | 0.0416239377424513 |

## Leave-Dataset Threshold Check

| action | target | mean_policy_uplift | fire_rate | precision_if_fire | always_fire_mean_uplift | positive_rate | features_used |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regime_tanh_vs_gelu | gain_tanh_all_vs_gelu | -0.05187108107877087 | 0.3333333333333333 | 0.0 | -0.11901814782161968 | 0.4444444444444444 | input_kurtosis;abstain;input_volatility |
| regime_softsign_vs_gelu | gain_softsign_all_vs_gelu | 0.0 | 0.0 | nan | -0.12714663915101776 | 0.3333333333333333 | input_volatility;abstain |
| regime_tanh_sin_vs_gelu | gain_tanh_sin001_all_vs_gelu | -0.05165496006111985 | 0.3333333333333333 | 0.0 | -0.11119379614786123 | 0.4444444444444444 | input_kurtosis;abstain;input_volatility |
| placement_encoder_tanh_vs_gelu | gain_enc_tanh_dec_gelu_vs_gelu | 0.0 | 0.0 | nan | -0.12351031609224163 | 0.3333333333333333 | input_volatility;abstain |
| placement_decoder_tanh_vs_gelu | gain_enc_gelu_dec_tanh_vs_gelu | -0.004085260911053611 | 1.0 | 0.8888888888888888 | -0.004085260911053611 | 0.8888888888888888 | input_volatility;input_volatility_shock |
| placement_output_tanh_vs_tanh | gain_outtanh_vs_tanh | -0.018304396942400342 | 0.3333333333333333 | 0.0 | -0.04678131897505056 | 0.3333333333333333 | input_kurtosis;input_mean;abstain |
| placement_all_tanh_vs_best_split | gain_alltanh_vs_best_split | -0.12849767731878436 | 1.0 | 0.3333333333333333 | -0.12849767731878436 | 0.3333333333333333 | input_kurtosis;router_freq_dominant_period_to_horizon_ratio;input_anomaly_density_intensity |
| oracle_gain_vs_gelu | oracle_gain_vs_gelu | 0.024035504463226345 | 1.0 | 1.0 | 0.024035504463226345 | 1.0 | router_freq_seasonal_peak_strength;input_anomaly_density_intensity |

## Router-Only Leave-Dataset Threshold Check

| action | target | mean_policy_uplift | fire_rate | precision_if_fire | always_fire_mean_uplift | positive_rate | features_used |
| --- | --- | --- | --- | --- | --- | --- | --- |
| regime_tanh_vs_gelu | gain_tanh_all_vs_gelu | -0.07693779202135874 | 0.3333333333333333 | 0.3333333333333333 | -0.11901814782161968 | 0.4444444444444444 | router_time_robust_scale_ratio;abstain |
| regime_softsign_vs_gelu | gain_softsign_all_vs_gelu | -0.08833891856737391 | 0.3333333333333333 | 0.0 | -0.12714663915101776 | 0.3333333333333333 | router_time_robust_scale_ratio;abstain |
| regime_tanh_sin_vs_gelu | gain_tanh_sin001_all_vs_gelu | -0.06954466071719834 | 0.3333333333333333 | 0.3333333333333333 | -0.11119379614786123 | 0.4444444444444444 | router_time_robust_scale_ratio;abstain |
| placement_encoder_tanh_vs_gelu | gain_enc_tanh_dec_gelu_vs_gelu | -0.06883745866659148 | 0.3333333333333333 | 0.0 | -0.12351031609224163 | 0.3333333333333333 | router_time_robust_scale_ratio;abstain |
| placement_decoder_tanh_vs_gelu | gain_enc_gelu_dec_tanh_vs_gelu | -0.012060701505883394 | 0.3333333333333333 | 0.6666666666666666 | -0.004085260911053611 | 0.8888888888888888 | router_time_robust_scale_ratio;router_time_late_early_level_shift |
| placement_output_tanh_vs_tanh | gain_outtanh_vs_tanh | -0.045742139333577986 | 0.6666666666666666 | 0.16666666666666666 | -0.04678131897505056 | 0.3333333333333333 | router_time_robust_scale_ratio;router_time_late_early_level_shift;abstain |
| placement_all_tanh_vs_best_split | gain_alltanh_vs_best_split | -0.0693607232837386 | 0.6666666666666666 | 0.5 | -0.12849767731878436 | 0.3333333333333333 | router_time_robust_scale_ratio;router_freq_dominant_period_to_horizon_ratio |
| oracle_gain_vs_gelu | oracle_gain_vs_gelu | 0.016820551944643432 | 0.6666666666666666 | 1.0 | 0.024035504463226345 | 1.0 | router_freq_seasonal_peak_strength;router_time_local_roughness_ratio |

## Top Spearman Correlations

| action | target | feature | spearman | abs_spearman |
| --- | --- | --- | --- | --- |
| oracle_gain_vs_gelu | oracle_gain_vs_gelu | pred_to_seq_ratio | -0.735284485661027 | 0.735284485661027 |
| oracle_gain_vs_gelu | oracle_gain_vs_gelu | router_freq_dominant_period_to_horizon_ratio | 0.7166666666666667 | 0.7166666666666667 |
| oracle_gain_vs_gelu | oracle_gain_vs_gelu | pred_len | -0.6582805886043832 | 0.6582805886043832 |
| oracle_gain_vs_gelu | oracle_gain_vs_gelu | router_timefreq_burstiness_index | -0.5499999999999999 | 0.5499999999999999 |
| placement_all_tanh_vs_best_split | gain_alltanh_vs_best_split | pred_to_seq_ratio | -0.803683042466704 | 0.803683042466704 |
| placement_all_tanh_vs_best_split | gain_alltanh_vs_best_split | router_freq_dominant_period_to_horizon_ratio | 0.7833333333333333 | 0.7833333333333333 |
| placement_all_tanh_vs_best_split | gain_alltanh_vs_best_split | pred_len | -0.7764335147641444 | 0.7764335147641444 |
| placement_all_tanh_vs_best_split | gain_alltanh_vs_best_split | input_direction_entropy | -0.6333333333333333 | 0.6333333333333333 |
| placement_decoder_tanh_vs_gelu | gain_enc_gelu_dec_tanh_vs_gelu | router_time_robust_scale_ratio | -0.7 | 0.7 |
| placement_decoder_tanh_vs_gelu | gain_enc_gelu_dec_tanh_vs_gelu | router_time_outlier_mass_z3 | -0.7 | 0.7 |
| placement_decoder_tanh_vs_gelu | gain_enc_gelu_dec_tanh_vs_gelu | router_time_iqr_mad_ratio | -0.6666666666666667 | 0.6666666666666667 |
| placement_decoder_tanh_vs_gelu | gain_enc_gelu_dec_tanh_vs_gelu | input_direction_entropy | 0.6333333333333333 | 0.6333333333333333 |
| placement_encoder_tanh_vs_gelu | gain_enc_tanh_dec_gelu_vs_gelu | pred_len | -0.8355099778440249 | 0.8355099778440249 |
| placement_encoder_tanh_vs_gelu | gain_enc_tanh_dec_gelu_vs_gelu | router_time_robust_scale_ratio | 0.7666666666666667 | 0.7666666666666667 |
| placement_encoder_tanh_vs_gelu | gain_enc_tanh_dec_gelu_vs_gelu | router_time_outlier_mass_z3 | 0.7666666666666667 | 0.7666666666666667 |
| placement_encoder_tanh_vs_gelu | gain_enc_tanh_dec_gelu_vs_gelu | input_lag1_autocorr | -0.75 | 0.75 |
| placement_output_tanh_vs_tanh | gain_outtanh_vs_tanh | router_timefreq_burstiness_index | 0.7999999999999999 | 0.7999999999999999 |
| placement_output_tanh_vs_tanh | gain_outtanh_vs_tanh | input_mean_crossing_rate | -0.75 | 0.75 |
| placement_output_tanh_vs_tanh | gain_outtanh_vs_tanh | input_lag1_autocorr | 0.7166666666666667 | 0.7166666666666667 |
| placement_output_tanh_vs_tanh | gain_outtanh_vs_tanh | input_low_freq_energy_ratio | 0.7166666666666667 | 0.7166666666666667 |
| regime_softsign_vs_gelu | gain_softsign_all_vs_gelu | pred_len | -0.8017519989412359 | 0.8017519989412359 |
| regime_softsign_vs_gelu | gain_softsign_all_vs_gelu | input_lag1_autocorr | -0.7833333333333333 | 0.7833333333333333 |
| regime_softsign_vs_gelu | gain_softsign_all_vs_gelu | input_low_freq_energy_ratio | -0.7833333333333333 | 0.7833333333333333 |
| regime_softsign_vs_gelu | gain_softsign_all_vs_gelu | input_mid_freq_energy_ratio | 0.7833333333333333 | 0.7833333333333333 |
| regime_tanh_sin_vs_gelu | gain_tanh_sin001_all_vs_gelu | router_freq_dominant_period_to_horizon_ratio | 0.75 | 0.75 |
| regime_tanh_sin_vs_gelu | gain_tanh_sin001_all_vs_gelu | pred_len | -0.7426755358613554 | 0.7426755358613554 |
| regime_tanh_sin_vs_gelu | gain_tanh_sin001_all_vs_gelu | pred_to_seq_ratio | -0.735284485661027 | 0.735284485661027 |
| regime_tanh_sin_vs_gelu | gain_tanh_sin001_all_vs_gelu | router_time_robust_scale_ratio | 0.6333333333333333 | 0.6333333333333333 |
| regime_tanh_vs_gelu | gain_tanh_all_vs_gelu | router_freq_dominant_period_to_horizon_ratio | 0.75 | 0.75 |
| regime_tanh_vs_gelu | gain_tanh_all_vs_gelu | pred_len | -0.7426755358613554 | 0.7426755358613554 |
| regime_tanh_vs_gelu | gain_tanh_all_vs_gelu | pred_to_seq_ratio | -0.735284485661027 | 0.735284485661027 |
| regime_tanh_vs_gelu | gain_tanh_all_vs_gelu | router_time_robust_scale_ratio | 0.6333333333333333 | 0.6333333333333333 |

## Best In-Sample Single-Feature Rules

| action | target | feature | direction | threshold | fired_count | fired_rate | precision | mean_uplift_if_fired | policy_mean_uplift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_gain_vs_gelu | oracle_gain_vs_gelu | router_freq_seasonal_peak_strength | high | 0.0 | 9 | 1.0 | 1.0 | 0.024035504463226345 | 0.024035504463226345 |
| regime_tanh_sin_vs_gelu | gain_tanh_sin001_all_vs_gelu | input_direction_entropy | low | 0.7712858702000919 | 4 | 0.4444444444444444 | 1.0 | 0.03469013095400028 | 0.015417835979555681 |
| placement_all_tanh_vs_best_split | gain_alltanh_vs_best_split | input_direction_entropy | low | 0.7712858702000919 | 4 | 0.4444444444444444 | 0.75 | 0.033252499599362204 | 0.014778888710827645 |
| regime_tanh_vs_gelu | gain_tanh_all_vs_gelu | input_direction_entropy | low | 0.7712858702000919 | 4 | 0.4444444444444444 | 1.0 | 0.03243511631226018 | 0.014415607249893412 |
| placement_decoder_tanh_vs_gelu | gain_enc_gelu_dec_tanh_vs_gelu | input_volatility | high | 0.018993869388280946 | 7 | 0.7777777777777778 | 1.0 | 0.012196573624398743 | 0.009486223930087911 |
| placement_encoder_tanh_vs_gelu | gain_enc_tanh_dec_gelu_vs_gelu | input_volatility | high | 0.04835579452581115 | 3 | 0.3333333333333333 | 1.0 | 0.026349652122663717 | 0.008783217374221238 |
| regime_softsign_vs_gelu | gain_softsign_all_vs_gelu | input_volatility | high | 0.04835579452581115 | 3 | 0.3333333333333333 | 1.0 | 0.02263557787510187 | 0.007545192625033957 |
| placement_output_tanh_vs_tanh | gain_outtanh_vs_tanh | input_kurtosis | high | -0.3368407267978427 | 2 | 0.2222222222222222 | 1.0 | 0.015951479112184194 | 0.0035447731360409318 |

## Best Router-Only In-Sample Single-Feature Rules

| action | target | feature | direction | threshold | fired_count | fired_rate | precision | mean_uplift_if_fired | policy_mean_uplift |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| oracle_gain_vs_gelu | oracle_gain_vs_gelu | router_freq_seasonal_peak_strength | high | 0.0 | 9 | 1.0 | 1.0 | 0.024035504463226345 | 0.024035504463226345 |
| regime_tanh_sin_vs_gelu | gain_tanh_sin001_all_vs_gelu | router_time_robust_scale_ratio | high | 1.629118158125354 | 4 | 0.4444444444444444 | 1.0 | 0.03469013095400028 | 0.015417835979555681 |
| placement_all_tanh_vs_best_split | gain_alltanh_vs_best_split | router_time_robust_scale_ratio | high | 1.629118158125354 | 4 | 0.4444444444444444 | 0.75 | 0.033252499599362204 | 0.014778888710827645 |
| regime_tanh_vs_gelu | gain_tanh_all_vs_gelu | router_time_robust_scale_ratio | high | 1.629118158125354 | 4 | 0.4444444444444444 | 1.0 | 0.03243511631226018 | 0.014415607249893412 |
| placement_decoder_tanh_vs_gelu | gain_enc_gelu_dec_tanh_vs_gelu | router_norm_normalization_drift_score | high | 1.0481195968036452 | 7 | 0.7777777777777778 | 1.0 | 0.012196573624398743 | 0.009486223930087911 |
| placement_encoder_tanh_vs_gelu | gain_enc_tanh_dec_gelu_vs_gelu | router_time_robust_scale_ratio | high | 1.656097497148294 | 3 | 0.3333333333333333 | 1.0 | 0.026349652122663717 | 0.008783217374221238 |
| regime_softsign_vs_gelu | gain_softsign_all_vs_gelu | router_time_robust_scale_ratio | high | 1.656097497148294 | 3 | 0.3333333333333333 | 1.0 | 0.02263557787510187 | 0.007545192625033957 |
| placement_output_tanh_vs_tanh | gain_outtanh_vs_tanh | router_time_local_roughness_ratio | low | 0.1754222466736892 | 2 | 0.2222222222222222 | 1.0 | 0.015951479112184194 | 0.0035447731360409318 |

## Factor Extraction Sanity

| dataset | input_feature_count | nonzero_input_features | zero_or_missing_input_feature_rate |
| --- | --- | --- | --- |
| Exchange | 39 | 39 | 0.0 |
| ILI | 39 | 39 | 0.0 |
| Weather | 39 | 39 | 0.0 |

## Cell Profile

| dataset | pred_len | oracle_config | oracle_gain_vs_gelu | input_volatility | input_mean_abs_change | input_lag1_autocorr | input_spectral_entropy | input_high_freq_energy_ratio | input_low_freq_energy_ratio | router_time_local_roughness_ratio | router_time_trend_to_noise_ratio | router_freq_spectral_entropy | router_freq_high_freq_noise_share | router_freq_seasonal_peak_strength | router_timefreq_burstiness_index | router_multivar_cross_channel_mean_abs_corr | router_norm_normalization_drift_score | pred_to_seq_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exchange | 96 | enc_gelu_dec_tanh | 0.03456457725671016 | 0.01931615317988082 | 0.0031587282774446657 | -0.04413354685261617 | 0.4504557944894024 | 0.03121192729965992 | 0.9053415349222033 | 0.24245154772868338 | 3.4061303464434856 | 0.45045290614037364 | 0.031276545944598624 | 0.0 | 3.6850084419123164 | 0.5026317227734328 | 2.2471978706734914 | 1.0 |
| Exchange | 192 | enc_gelu_dec_tanh | 0.014444955181383286 | 0.0191534119622163 | 0.0031373138128125734 | -0.04367493579826387 | 0.4503566436321277 | 0.03125006307514517 | 0.9055350563933237 | 0.2421575940338646 | 3.416498837122651 | 0.45049807062222647 | 0.031133466161853245 | 0.0 | 3.6986030498417435 | 0.5045911936802594 | 2.2926735220817203 | 2.0 |
| Exchange | 336 | enc_gelu_dec_tanh | 0.007583739654931813 | 0.01923998187014256 | 0.0030802928577789643 | -0.032093020293009075 | 0.44605977140578806 | 0.03020124691948193 | 0.9071789034027885 | 0.2353325085821768 | 3.549468238397819 | 0.44622396534466013 | 0.030287244377680746 | 0.0 | 3.8062161405520065 | 0.4971814424374933 | 2.4101576136821814 | 3.5 |
| ILI | 24 | tanh_sin001_all | 0.02059202609431526 | 0.174062274701074 | 48559.935294117655 | -0.10486861796291251 | 0.41818533145335335 | 0.0588073195286595 | 0.8483439588009904 | 0.3189083285589343 | 2.9856655522528466 | 0.41818526918462945 | 0.0588073195286595 | 0.0 | 3.7213100564125265 | 0.8240323133765397 | 2.6518234251858233 | 0.6666666666666666 |
| ILI | 36 | enc_tanh_dec_gelu | 0.04013335431990009 | 0.1705486367285726 | 47953.9582278481 | -0.15214951690149695 | 0.4254154336136358 | 0.06275445655418732 | 0.8400130490919271 | 0.32721558444888044 | 3.1515522648331085 | 0.4254153713448783 | 0.06275445655418732 | 0.0 | 3.680797263236007 | 0.8359088272646226 | 2.606748295932966 | 1.0 |
| ILI | 48 | tanh_all | 0.0340663149945503 | 0.16451435990953298 | 46004.969667318976 | -0.18156677481901326 | 0.43242964189977884 | 0.0664594460455802 | 0.8314644964049589 | 0.3334056388460193 | 3.2006700055014967 | 0.43242957963104833 | 0.0664594460455802 | 0.0 | 3.525526851660988 | 0.8451113204062815 | 2.5374414916611645 | 1.3333333333333333 |
| Weather | 96 | tanh_sin001_all | 0.04870810214188847 | 0.018873523909993795 | 1.1569589257927089 | 0.1461896206732654 | 0.3392067630068098 | 0.015627502987366204 | 0.9508589313683354 | 0.17632747187743536 | 3.7401519766927365 | 0.36261416718596295 | 0.020898078548419696 | 0.7563533134190517 | 4.259600217248858 | 0.5390825404739044 | 1.0479706126376285 | 1.0 |
| Weather | 192 | enc_gelu_dec_tanh | 0.0026294205080345523 | 0.01897539873420747 | 1.1593822393300588 | 0.14645024952509836 | 0.33782725757188614 | 0.01558366017588814 | 0.9511206991895474 | 0.17519594037275266 | 3.762387333870837 | 0.36142829559189005 | 0.021144347010484427 | 0.7585999965415176 | 4.3192363421125854 | 0.5384657416920542 | 1.048081355766911 | 2.0 |
| Weather | 336 | enc_gelu_dec_tanh | 0.013597050017323163 | 0.019067752004574848 | 1.1614912087128906 | 0.14851656579076958 | 0.3358799695030777 | 0.01543582815629946 | 0.9515997374019354 | 0.1739451239610314 | 3.776498225722964 | 0.3599539573688044 | 0.020926355724856197 | 0.7605460544041867 | 4.288464830138194 | 0.5385172146884867 | 1.0482725609505814 | 3.5 |

## Interpretation Boundary

- The sample size is 9 dataset-horizon cells, so high in-sample correlations are hypothesis-generating only.
- The current signal supports coarse abstain/calibration gates better than exact zero-shot config selection.
- input_* factors pass the granularity sanity check for this audit; no dataset has material zero/missing collapse.
- Router-only leave-dataset checks are the more conservative read of generalization because they remove dependence on the input_* extraction path.
