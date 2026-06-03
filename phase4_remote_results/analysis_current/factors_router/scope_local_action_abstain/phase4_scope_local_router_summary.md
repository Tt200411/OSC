# Scope-Local Two-Stage Router Summary

- Cells: 57
- Top-k feature candidates per set: 6
- Evaluation modes: leave_cell, leave_dataset

This protocol first gates by model scope, then learns a data-factor
refinement rule inside each seen scope. The static comparator is also
selected from the training fold for the same scope.

| cell_count | mean_gain_vs_baseline | mean_gain_vs_train_static | positive_vs_train_static_rate | mean_regret_vs_oracle | config_hit_rate | action_hit_rate | dynamic_use_rate | eval_name | feature_set | target_kind | abstain | variant | top_features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 57 | 0.07362460053515878 | 0.0005721949633009493 | 0.07017543859649122 | 0.03887524693035972 | 0.3508771929824561 | 0.7719298245614035 | 0.07017543859649122 | leave_cell | frequency_only | action | True | frequency_only:action:abstain | input_spectral_entropy;input_dominant_frequency_energy_ratio;input_top3_frequency_energy_ratio;input_low_freq_energy_ratio;input_mid_freq_energy_ratio;input_high_freq_energy_ratio |
| 57 | 0.07284650150253466 | -0.0002842617431502151 | 0.07017543859649122 | 0.03974729123037617 | 0.3508771929824561 | 0.7543859649122807 | 0.22807017543859648 | leave_cell | data_only | action | True | data_only:action:abstain | input_volatility;input_anomaly_density_intensity;input_turbulence_score;input_mean;input_std;input_cv |
| 57 | 0.07284650150253466 | -0.0002842617431502151 | 0.07017543859649122 | 0.03974729123037617 | 0.3508771929824561 | 0.7543859649122807 | 0.22807017543859648 | leave_cell | statistical_only | action | True | statistical_only:action:abstain | input_volatility;input_anomaly_density_intensity;input_turbulence_score;input_mean;input_std;input_cv |
| 57 | 0.07409337181654005 | -0.0005327015321021042 | 0.0 | 0.037275666706393686 | 0.24561403508771928 | 0.6666666666666666 | 0.03508771929824561 | leave_dataset | frequency_only | action | True | frequency_only:action:abstain | input_spectral_entropy;input_dominant_frequency_energy_ratio;input_top3_frequency_energy_ratio;input_low_freq_energy_ratio;input_mid_freq_energy_ratio;input_high_freq_energy_ratio |
| 57 | 0.07075501402901033 | -0.004192020027824546 | 0.017543859649122806 | 0.04096393700893878 | 0.2631578947368421 | 0.631578947368421 | 0.17543859649122806 | leave_dataset | data_only | action | True | data_only:action:abstain | input_volatility;input_anomaly_density_intensity;input_turbulence_score;input_mean;input_std;input_cv |
| 57 | 0.07075501402901033 | -0.004192020027824546 | 0.017543859649122806 | 0.04096393700893878 | 0.2631578947368421 | 0.631578947368421 | 0.17543859649122806 | leave_dataset | statistical_only | action | True | statistical_only:action:abstain | input_volatility;input_anomaly_density_intensity;input_turbulence_score;input_mean;input_std;input_cv |
