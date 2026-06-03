# Phase-5 Router Oracle Evaluation

Generated: 2026-05-31T18:53:40.608771+00:00

## Evidence Status

- Expected seed-runs: 63
- Observed deduped seed-runs: 63
- Missing seed-runs: 0
- Aggregated dataset-horizon-config cells: 63
- Dataset-horizon oracle cells: 9

## Oracle Cells

| dataset | pred_len | gelu_mse | new_best_static_config | new_best_static_mse | oracle_config | oracle_mse | oracle_gain_vs_gelu |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exchange | 96 | 0.9700721 | softsign_all | 1.0052762 | enc_gelu_dec_tanh | 0.9305318 | 0.04076016617733875 |
| Exchange | 192 | 1.1812613 | softsign_all | 1.6034241 | enc_gelu_dec_tanh | 1.1411182 | 0.033983251631116755 |
| Exchange | 336 | 1.6637204 | softsign_all | 1.7164084 | gelu_all | 1.6637204 | 0.0 |
| ILI | 24 | 5.897227 | tanh_sin001_all | 5.8858104 | enc_gelu_dec_tanh | 5.853008 | 0.007498269949588172 |
| ILI | 36 | 5.681908 | tanh_sin001_all | 5.4969783 | enc_tanh_dec_gelu | 5.423634 | 0.045455505439370034 |
| ILI | 48 | 5.1766324 | tanh_sin001_all | 4.8986616 | tanh_sin001_all | 4.8986616 | 0.05369722601898491 |
| Weather | 96 | 0.41913968 | tanh_sin001_all | 0.33686465 | tanh_sin001_all | 0.33686465 | 0.19629501554231282 |
| Weather | 192 | 0.37179664 | tanh_sin001_all | 0.4940928 | gelu_all | 0.37179664 | 0.0 |
| Weather | 336 | 0.46428803 | tanh_all | 0.6539041 | gelu_all | 0.46428803 | 0.0 |

## Zero-Shot Router Summary

| feature_set | fallback_policy | cell_count | mean_gain_vs_gelu | median_gain_vs_gelu | mean_gain_vs_train_static | mean_gain_vs_new_static | positive_vs_gelu_rate | positive_vs_train_static_rate | positive_vs_new_static_rate | mean_regret_vs_oracle | median_regret_vs_oracle | mean_oracle_capture_ratio | config_hit_rate | action_hit_rate | abstain_rate | top_chosen_configs | top_oracle_configs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_input | gelu | 9 | -0.10499338823288033 | -0.04337522953190808 | 0.0 | -0.006569603268544218 | 0.4444444444444444 | 0.0 | 0.0 | 0.14899585051255868 | 0.069007027863576 | -1.480843473018133 | 0.2222222222222222 | 0.6666666666666666 | 0.0 | tanh_sin001_all | gelu_all;enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu |
| all_input | train_static | 9 | -0.10499338823288033 | -0.04337522953190808 | 0.0 | -0.006569603268544218 | 0.4444444444444444 | 0.0 | 0.0 | 0.14899585051255868 | 0.069007027863576 | -1.480843473018133 | 0.2222222222222222 | 0.6666666666666666 | 0.0 | tanh_sin001_all | gelu_all;enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu |
| all_with_topology | gelu | 9 | -0.10499338823288033 | -0.04337522953190808 | 0.0 | -0.006569603268544218 | 0.4444444444444444 | 0.0 | 0.0 | 0.14899585051255868 | 0.069007027863576 | -1.480843473018133 | 0.2222222222222222 | 0.6666666666666666 | 0.0 | tanh_sin001_all | gelu_all;enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu |
| all_with_topology | train_static | 9 | -0.10499338823288033 | -0.04337522953190808 | 0.0 | -0.006569603268544218 | 0.4444444444444444 | 0.0 | 0.0 | 0.14899585051255868 | 0.069007027863576 | -1.480843473018133 | 0.2222222222222222 | 0.6666666666666666 | 0.0 | tanh_sin001_all | gelu_all;enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu |
| data | gelu | 9 | -0.10499338823288033 | -0.04337522953190808 | 0.0 | -0.006569603268544218 | 0.4444444444444444 | 0.0 | 0.0 | 0.14899585051255868 | 0.069007027863576 | -1.480843473018133 | 0.2222222222222222 | 0.6666666666666666 | 0.0 | tanh_sin001_all | gelu_all;enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu |
| data | train_static | 9 | -0.10499338823288033 | -0.04337522953190808 | 0.0 | -0.006569603268544218 | 0.4444444444444444 | 0.0 | 0.0 | 0.14899585051255868 | 0.069007027863576 | -1.480843473018133 | 0.2222222222222222 | 0.6666666666666666 | 0.0 | tanh_sin001_all | gelu_all;enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu |
| statistical | gelu | 9 | -0.10499338823288033 | -0.04337522953190808 | 0.0 | -0.006569603268544218 | 0.4444444444444444 | 0.0 | 0.0 | 0.14899585051255868 | 0.069007027863576 | -1.480843473018133 | 0.2222222222222222 | 0.6666666666666666 | 0.0 | tanh_sin001_all | gelu_all;enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu |
| statistical | train_static | 9 | -0.10499338823288033 | -0.04337522953190808 | 0.0 | -0.006569603268544218 | 0.4444444444444444 | 0.0 | 0.0 | 0.14899585051255868 | 0.069007027863576 | -1.480843473018133 | 0.2222222222222222 | 0.6666666666666666 | 0.0 | tanh_sin001_all | gelu_all;enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu |
| topology | gelu | 9 | -0.10499338823288033 | -0.04337522953190808 | 0.0 | -0.006569603268544218 | 0.4444444444444444 | 0.0 | 0.0 | 0.14899585051255868 | 0.069007027863576 | -1.480843473018133 | 0.2222222222222222 | 0.6666666666666666 | 0.0 | tanh_sin001_all | gelu_all;enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu |
| topology | train_static | 9 | -0.10499338823288033 | -0.04337522953190808 | 0.0 | -0.006569603268544218 | 0.4444444444444444 | 0.0 | 0.0 | 0.14899585051255868 | 0.069007027863576 | -1.480843473018133 | 0.2222222222222222 | 0.6666666666666666 | 0.0 | tanh_sin001_all | gelu_all;enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu |
| frequency | gelu | 9 | -0.1060682114536155 | -0.0455712518688044 | -0.001135578871536806 | -0.007707601590621459 | 0.3333333333333333 | 0.1111111111111111 | 0.0 | 0.1501026528826747 | 0.06949569170396659 | -1.5921725548883015 | 0.1111111111111111 | 0.6666666666666666 | 0.0 | tanh_all;tanh_sin001_all | gelu_all;enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu |
| frequency | train_static | 9 | -0.1060682114536155 | -0.0455712518688044 | -0.001135578871536806 | -0.007707601590621459 | 0.3333333333333333 | 0.1111111111111111 | 0.0 | 0.1501026528826747 | 0.06949569170396659 | -1.5921725548883015 | 0.1111111111111111 | 0.6666666666666666 | 0.0 | tanh_all;tanh_sin001_all | gelu_all;enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu |

## By Dataset

| feature_set | fallback_policy | dataset | cell_count | mean_gain_vs_gelu | mean_gain_vs_train_static | mean_gain_vs_new_static | mean_regret_vs_oracle | config_hit_rate | action_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_input | gelu | Exchange | 3 | -0.15974504299427625 | 0.0 | -0.01666850669900485 | 0.19055151684345162 | 0.0 | 0.6666666666666666 |
| all_input | gelu | ILI | 3 | 0.029393421849664054 | 0.0 | 0.0 | 0.0063758192927578896 | 0.3333333333333333 | 1.0 |
| all_input | gelu | Weather | 3 | -0.18462854355402883 | 0.0 | -0.0030403031066278045 | 0.25006021540146645 | 0.3333333333333333 | 0.3333333333333333 |
| all_input | train_static | Exchange | 3 | -0.15974504299427625 | 0.0 | -0.01666850669900485 | 0.19055151684345162 | 0.0 | 0.6666666666666666 |
| all_input | train_static | ILI | 3 | 0.029393421849664054 | 0.0 | 0.0 | 0.0063758192927578896 | 0.3333333333333333 | 1.0 |
| all_input | train_static | Weather | 3 | -0.18462854355402883 | 0.0 | -0.0030403031066278045 | 0.25006021540146645 | 0.3333333333333333 | 0.3333333333333333 |
| all_with_topology | gelu | Exchange | 3 | -0.15974504299427625 | 0.0 | -0.01666850669900485 | 0.19055151684345162 | 0.0 | 0.6666666666666666 |
| all_with_topology | gelu | ILI | 3 | 0.029393421849664054 | 0.0 | 0.0 | 0.0063758192927578896 | 0.3333333333333333 | 1.0 |
| all_with_topology | gelu | Weather | 3 | -0.18462854355402883 | 0.0 | -0.0030403031066278045 | 0.25006021540146645 | 0.3333333333333333 | 0.3333333333333333 |
| all_with_topology | train_static | Exchange | 3 | -0.15974504299427625 | 0.0 | -0.01666850669900485 | 0.19055151684345162 | 0.0 | 0.6666666666666666 |
| all_with_topology | train_static | ILI | 3 | 0.029393421849664054 | 0.0 | 0.0 | 0.0063758192927578896 | 0.3333333333333333 | 1.0 |
| all_with_topology | train_static | Weather | 3 | -0.18462854355402883 | 0.0 | -0.0030403031066278045 | 0.25006021540146645 | 0.3333333333333333 | 0.3333333333333333 |
| data | gelu | Exchange | 3 | -0.15974504299427625 | 0.0 | -0.01666850669900485 | 0.19055151684345162 | 0.0 | 0.6666666666666666 |
| data | gelu | ILI | 3 | 0.029393421849664054 | 0.0 | 0.0 | 0.0063758192927578896 | 0.3333333333333333 | 1.0 |
| data | gelu | Weather | 3 | -0.18462854355402883 | 0.0 | -0.0030403031066278045 | 0.25006021540146645 | 0.3333333333333333 | 0.3333333333333333 |
| data | train_static | Exchange | 3 | -0.15974504299427625 | 0.0 | -0.01666850669900485 | 0.19055151684345162 | 0.0 | 0.6666666666666666 |
| data | train_static | ILI | 3 | 0.029393421849664054 | 0.0 | 0.0 | 0.0063758192927578896 | 0.3333333333333333 | 1.0 |
| data | train_static | Weather | 3 | -0.18462854355402883 | 0.0 | -0.0030403031066278045 | 0.25006021540146645 | 0.3333333333333333 | 0.3333333333333333 |
| frequency | gelu | Exchange | 3 | -0.16004179243316669 | -0.00041634139001189754 | -0.017092106440638053 | 0.19085832883428355 | 0.0 | 0.6666666666666666 |
| frequency | gelu | ILI | 3 | 0.026465701626348967 | -0.00299039522459852 | -0.00299039522459852 | 0.009389414412274076 | 0.0 | 1.0 |
| frequency | gelu | Weather | 3 | -0.18462854355402883 | 0.0 | -0.0030403031066278045 | 0.25006021540146645 | 0.3333333333333333 | 0.3333333333333333 |
| frequency | train_static | Exchange | 3 | -0.16004179243316669 | -0.00041634139001189754 | -0.017092106440638053 | 0.19085832883428355 | 0.0 | 0.6666666666666666 |
| frequency | train_static | ILI | 3 | 0.026465701626348967 | -0.00299039522459852 | -0.00299039522459852 | 0.009389414412274076 | 0.0 | 1.0 |
| frequency | train_static | Weather | 3 | -0.18462854355402883 | 0.0 | -0.0030403031066278045 | 0.25006021540146645 | 0.3333333333333333 | 0.3333333333333333 |
| statistical | gelu | Exchange | 3 | -0.15974504299427625 | 0.0 | -0.01666850669900485 | 0.19055151684345162 | 0.0 | 0.6666666666666666 |
| statistical | gelu | ILI | 3 | 0.029393421849664054 | 0.0 | 0.0 | 0.0063758192927578896 | 0.3333333333333333 | 1.0 |
| statistical | gelu | Weather | 3 | -0.18462854355402883 | 0.0 | -0.0030403031066278045 | 0.25006021540146645 | 0.3333333333333333 | 0.3333333333333333 |
| statistical | train_static | Exchange | 3 | -0.15974504299427625 | 0.0 | -0.01666850669900485 | 0.19055151684345162 | 0.0 | 0.6666666666666666 |
| statistical | train_static | ILI | 3 | 0.029393421849664054 | 0.0 | 0.0 | 0.0063758192927578896 | 0.3333333333333333 | 1.0 |
| statistical | train_static | Weather | 3 | -0.18462854355402883 | 0.0 | -0.0030403031066278045 | 0.25006021540146645 | 0.3333333333333333 | 0.3333333333333333 |
| topology | gelu | Exchange | 3 | -0.15974504299427625 | 0.0 | -0.01666850669900485 | 0.19055151684345162 | 0.0 | 0.6666666666666666 |
| topology | gelu | ILI | 3 | 0.029393421849664054 | 0.0 | 0.0 | 0.0063758192927578896 | 0.3333333333333333 | 1.0 |
| topology | gelu | Weather | 3 | -0.18462854355402883 | 0.0 | -0.0030403031066278045 | 0.25006021540146645 | 0.3333333333333333 | 0.3333333333333333 |
| topology | train_static | Exchange | 3 | -0.15974504299427625 | 0.0 | -0.01666850669900485 | 0.19055151684345162 | 0.0 | 0.6666666666666666 |
| topology | train_static | ILI | 3 | 0.029393421849664054 | 0.0 | 0.0 | 0.0063758192927578896 | 0.3333333333333333 | 1.0 |
| topology | train_static | Weather | 3 | -0.18462854355402883 | 0.0 | -0.0030403031066278045 | 0.25006021540146645 | 0.3333333333333333 | 0.3333333333333333 |

## Cell-Level Router Choices

| feature_set | fallback_policy | dataset | pred_len | chosen_config | oracle_config | gain_vs_gelu | gain_vs_train_static | gain_vs_new_static | regret_vs_oracle | config_hit | abstained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_input | gelu | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.19629501554231282 | 0.0 | 0.0 | 0.0 | True | False |
| all_input | train_static | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.19629501554231282 | 0.0 | 0.0 | 0.0 | True | False |
| all_input | gelu | Weather | 192 | tanh_sin001_all | gelu_all | -0.3289329349506762 | 0.0 | 0.0 | 0.3289329349506762 | False | False |
| all_input | train_static | Weather | 192 | tanh_sin001_all | gelu_all | -0.3289329349506762 | 0.0 | 0.0 | 0.3289329349506762 | False | False |
| all_input | gelu | Weather | 336 | tanh_sin001_all | gelu_all | -0.4212477112537231 | 0.0 | -0.009120909319883413 | 0.4212477112537231 | False | False |
| all_input | train_static | Weather | 336 | tanh_sin001_all | gelu_all | -0.4212477112537231 | 0.0 | -0.009120909319883413 | 0.4212477112537231 | False | False |
| all_input | gelu | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.04337522953190808 | 0.0 | -0.0068369270057324835 | 0.08771048985107233 | False | False |
| all_input | train_static | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.04337522953190808 | 0.0 | -0.0068369270057324835 | 0.08771048985107233 | False | False |
| all_input | gelu | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.36685287158734464 | 0.0 | -0.006976507338264325 | 0.4149370328157065 | False | False |
| all_input | train_static | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.36685287158734464 | 0.0 | -0.006976507338264325 | 0.4149370328157065 | False | False |
| all_input | gelu | Exchange | 336 | tanh_sin001_all | gelu_all | -0.069007027863576 | 0.0 | -0.03619208575301774 | 0.069007027863576 | False | False |
| all_input | train_static | Exchange | 336 | tanh_sin001_all | gelu_all | -0.069007027863576 | 0.0 | -0.03619208575301774 | 0.069007027863576 | False | False |
| all_input | gelu | ILI | 24 | tanh_sin001_all | enc_gelu_dec_tanh | 0.0019359268347648065 | 0.0 | 0.0 | 0.005604366165226573 | False | False |
| all_input | train_static | ILI | 24 | tanh_sin001_all | enc_gelu_dec_tanh | 0.0019359268347648065 | 0.0 | 0.0 | 0.005604366165226573 | False | False |
| all_input | gelu | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03254711269524245 | 0.0 | 0.0 | 0.013523091713047096 | False | False |
| all_input | train_static | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03254711269524245 | 0.0 | 0.0 | 0.013523091713047096 | False | False |
| all_input | gelu | ILI | 48 | tanh_sin001_all | tanh_sin001_all | 0.05369722601898491 | 0.0 | 0.0 | 0.0 | True | False |
| all_input | train_static | ILI | 48 | tanh_sin001_all | tanh_sin001_all | 0.05369722601898491 | 0.0 | 0.0 | 0.0 | True | False |
| all_with_topology | gelu | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.19629501554231282 | 0.0 | 0.0 | 0.0 | True | False |
| all_with_topology | train_static | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.19629501554231282 | 0.0 | 0.0 | 0.0 | True | False |
| all_with_topology | gelu | Weather | 192 | tanh_sin001_all | gelu_all | -0.3289329349506762 | 0.0 | 0.0 | 0.3289329349506762 | False | False |
| all_with_topology | train_static | Weather | 192 | tanh_sin001_all | gelu_all | -0.3289329349506762 | 0.0 | 0.0 | 0.3289329349506762 | False | False |
| all_with_topology | gelu | Weather | 336 | tanh_sin001_all | gelu_all | -0.4212477112537231 | 0.0 | -0.009120909319883413 | 0.4212477112537231 | False | False |
| all_with_topology | train_static | Weather | 336 | tanh_sin001_all | gelu_all | -0.4212477112537231 | 0.0 | -0.009120909319883413 | 0.4212477112537231 | False | False |
| all_with_topology | gelu | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.04337522953190808 | 0.0 | -0.0068369270057324835 | 0.08771048985107233 | False | False |
| all_with_topology | train_static | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.04337522953190808 | 0.0 | -0.0068369270057324835 | 0.08771048985107233 | False | False |
| all_with_topology | gelu | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.36685287158734464 | 0.0 | -0.006976507338264325 | 0.4149370328157065 | False | False |
| all_with_topology | train_static | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.36685287158734464 | 0.0 | -0.006976507338264325 | 0.4149370328157065 | False | False |
| all_with_topology | gelu | Exchange | 336 | tanh_sin001_all | gelu_all | -0.069007027863576 | 0.0 | -0.03619208575301774 | 0.069007027863576 | False | False |
| all_with_topology | train_static | Exchange | 336 | tanh_sin001_all | gelu_all | -0.069007027863576 | 0.0 | -0.03619208575301774 | 0.069007027863576 | False | False |
| all_with_topology | gelu | ILI | 24 | tanh_sin001_all | enc_gelu_dec_tanh | 0.0019359268347648065 | 0.0 | 0.0 | 0.005604366165226573 | False | False |
| all_with_topology | train_static | ILI | 24 | tanh_sin001_all | enc_gelu_dec_tanh | 0.0019359268347648065 | 0.0 | 0.0 | 0.005604366165226573 | False | False |
| all_with_topology | gelu | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03254711269524245 | 0.0 | 0.0 | 0.013523091713047096 | False | False |
| all_with_topology | train_static | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03254711269524245 | 0.0 | 0.0 | 0.013523091713047096 | False | False |
| all_with_topology | gelu | ILI | 48 | tanh_sin001_all | tanh_sin001_all | 0.05369722601898491 | 0.0 | 0.0 | 0.0 | True | False |
| all_with_topology | train_static | ILI | 48 | tanh_sin001_all | tanh_sin001_all | 0.05369722601898491 | 0.0 | 0.0 | 0.0 | True | False |
| data | gelu | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.19629501554231282 | 0.0 | 0.0 | 0.0 | True | False |
| data | train_static | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.19629501554231282 | 0.0 | 0.0 | 0.0 | True | False |
| data | gelu | Weather | 192 | tanh_sin001_all | gelu_all | -0.3289329349506762 | 0.0 | 0.0 | 0.3289329349506762 | False | False |
| data | train_static | Weather | 192 | tanh_sin001_all | gelu_all | -0.3289329349506762 | 0.0 | 0.0 | 0.3289329349506762 | False | False |
| data | gelu | Weather | 336 | tanh_sin001_all | gelu_all | -0.4212477112537231 | 0.0 | -0.009120909319883413 | 0.4212477112537231 | False | False |
| data | train_static | Weather | 336 | tanh_sin001_all | gelu_all | -0.4212477112537231 | 0.0 | -0.009120909319883413 | 0.4212477112537231 | False | False |
| data | gelu | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.04337522953190808 | 0.0 | -0.0068369270057324835 | 0.08771048985107233 | False | False |
| data | train_static | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.04337522953190808 | 0.0 | -0.0068369270057324835 | 0.08771048985107233 | False | False |
| data | gelu | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.36685287158734464 | 0.0 | -0.006976507338264325 | 0.4149370328157065 | False | False |
| data | train_static | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.36685287158734464 | 0.0 | -0.006976507338264325 | 0.4149370328157065 | False | False |
| data | gelu | Exchange | 336 | tanh_sin001_all | gelu_all | -0.069007027863576 | 0.0 | -0.03619208575301774 | 0.069007027863576 | False | False |
| data | train_static | Exchange | 336 | tanh_sin001_all | gelu_all | -0.069007027863576 | 0.0 | -0.03619208575301774 | 0.069007027863576 | False | False |
| data | gelu | ILI | 24 | tanh_sin001_all | enc_gelu_dec_tanh | 0.0019359268347648065 | 0.0 | 0.0 | 0.005604366165226573 | False | False |
| data | train_static | ILI | 24 | tanh_sin001_all | enc_gelu_dec_tanh | 0.0019359268347648065 | 0.0 | 0.0 | 0.005604366165226573 | False | False |
| data | gelu | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03254711269524245 | 0.0 | 0.0 | 0.013523091713047096 | False | False |
| data | train_static | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03254711269524245 | 0.0 | 0.0 | 0.013523091713047096 | False | False |
| data | gelu | ILI | 48 | tanh_sin001_all | tanh_sin001_all | 0.05369722601898491 | 0.0 | 0.0 | 0.0 | True | False |
| data | train_static | ILI | 48 | tanh_sin001_all | tanh_sin001_all | 0.05369722601898491 | 0.0 | 0.0 | 0.0 | True | False |
| frequency | gelu | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.19629501554231282 | 0.0 | 0.0 | 0.0 | True | False |
| frequency | train_static | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.19629501554231282 | 0.0 | 0.0 | 0.0 | True | False |
| frequency | gelu | Weather | 192 | tanh_sin001_all | gelu_all | -0.3289329349506762 | 0.0 | 0.0 | 0.3289329349506762 | False | False |
| frequency | train_static | Weather | 192 | tanh_sin001_all | gelu_all | -0.3289329349506762 | 0.0 | 0.0 | 0.3289329349506762 | False | False |
| frequency | gelu | Weather | 336 | tanh_sin001_all | gelu_all | -0.4212477112537231 | 0.0 | -0.009120909319883413 | 0.4212477112537231 | False | False |
| frequency | train_static | Weather | 336 | tanh_sin001_all | gelu_all | -0.4212477112537231 | 0.0 | -0.009120909319883413 | 0.4212477112537231 | False | False |
| frequency | gelu | Exchange | 96 | tanh_all | enc_gelu_dec_tanh | -0.0455712518688044 | -0.002104729223715158 | -0.008956046109517014 | 0.0899998259060034 | False | False |
| frequency | train_static | Exchange | 96 | tanh_all | enc_gelu_dec_tanh | -0.0455712518688044 | -0.002104729223715158 | -0.008956046109517014 | 0.0899998259060034 | False | False |
| frequency | gelu | Exchange | 192 | tanh_all | enc_gelu_dec_tanh | -0.3650584337267291 | 0.0013128244435932947 | -0.005654523965306449 | 0.41307946889288066 | False | False |
| frequency | train_static | Exchange | 192 | tanh_all | enc_gelu_dec_tanh | -0.3650584337267291 | 0.0013128244435932947 | -0.005654523965306449 | 0.41307946889288066 | False | False |
| frequency | gelu | Exchange | 336 | tanh_all | gelu_all | -0.06949569170396659 | -0.00045711938991382917 | -0.03666574924709069 | 0.06949569170396659 | False | False |
| frequency | train_static | Exchange | 336 | tanh_all | gelu_all | -0.06949569170396659 | -0.00045711938991382917 | -0.03666574924709069 | 0.06949569170396659 | False | False |
| frequency | gelu | ILI | 24 | tanh_all | enc_gelu_dec_tanh | -0.0023549542861415586 | -0.004299204065424751 | -0.004299204065424751 | 0.009927664544452994 | False | False |
| frequency | train_static | ILI | 24 | tanh_all | enc_gelu_dec_tanh | -0.0023549542861415586 | -0.004299204065424751 | -0.004299204065424751 | 0.009927664544452994 | False | False |
| frequency | gelu | ILI | 36 | tanh_all | enc_tanh_dec_gelu | 0.029291621758043302 | -0.003365012374161879 | -0.003365012374161879 | 0.016933609458160304 | False | False |
| frequency | train_static | ILI | 36 | tanh_all | enc_tanh_dec_gelu | 0.029291621758043302 | -0.003365012374161879 | -0.003365012374161879 | 0.016933609458160304 | False | False |
| frequency | gelu | ILI | 48 | tanh_all | tanh_sin001_all | 0.052460437407145157 | -0.001306969234208931 | -0.001306969234208931 | 0.001306969234208931 | False | False |
| frequency | train_static | ILI | 48 | tanh_all | tanh_sin001_all | 0.052460437407145157 | -0.001306969234208931 | -0.001306969234208931 | 0.001306969234208931 | False | False |
| statistical | gelu | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.19629501554231282 | 0.0 | 0.0 | 0.0 | True | False |
| statistical | train_static | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.19629501554231282 | 0.0 | 0.0 | 0.0 | True | False |
| statistical | gelu | Weather | 192 | tanh_sin001_all | gelu_all | -0.3289329349506762 | 0.0 | 0.0 | 0.3289329349506762 | False | False |
| statistical | train_static | Weather | 192 | tanh_sin001_all | gelu_all | -0.3289329349506762 | 0.0 | 0.0 | 0.3289329349506762 | False | False |
| statistical | gelu | Weather | 336 | tanh_sin001_all | gelu_all | -0.4212477112537231 | 0.0 | -0.009120909319883413 | 0.4212477112537231 | False | False |
| statistical | train_static | Weather | 336 | tanh_sin001_all | gelu_all | -0.4212477112537231 | 0.0 | -0.009120909319883413 | 0.4212477112537231 | False | False |
| statistical | gelu | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.04337522953190808 | 0.0 | -0.0068369270057324835 | 0.08771048985107233 | False | False |
| statistical | train_static | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.04337522953190808 | 0.0 | -0.0068369270057324835 | 0.08771048985107233 | False | False |
| statistical | gelu | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.36685287158734464 | 0.0 | -0.006976507338264325 | 0.4149370328157065 | False | False |
| statistical | train_static | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.36685287158734464 | 0.0 | -0.006976507338264325 | 0.4149370328157065 | False | False |
| statistical | gelu | Exchange | 336 | tanh_sin001_all | gelu_all | -0.069007027863576 | 0.0 | -0.03619208575301774 | 0.069007027863576 | False | False |
| statistical | train_static | Exchange | 336 | tanh_sin001_all | gelu_all | -0.069007027863576 | 0.0 | -0.03619208575301774 | 0.069007027863576 | False | False |
| statistical | gelu | ILI | 24 | tanh_sin001_all | enc_gelu_dec_tanh | 0.0019359268347648065 | 0.0 | 0.0 | 0.005604366165226573 | False | False |
| statistical | train_static | ILI | 24 | tanh_sin001_all | enc_gelu_dec_tanh | 0.0019359268347648065 | 0.0 | 0.0 | 0.005604366165226573 | False | False |
| statistical | gelu | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03254711269524245 | 0.0 | 0.0 | 0.013523091713047096 | False | False |
| statistical | train_static | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03254711269524245 | 0.0 | 0.0 | 0.013523091713047096 | False | False |
| statistical | gelu | ILI | 48 | tanh_sin001_all | tanh_sin001_all | 0.05369722601898491 | 0.0 | 0.0 | 0.0 | True | False |
| statistical | train_static | ILI | 48 | tanh_sin001_all | tanh_sin001_all | 0.05369722601898491 | 0.0 | 0.0 | 0.0 | True | False |
| topology | gelu | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.19629501554231282 | 0.0 | 0.0 | 0.0 | True | False |
| topology | train_static | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.19629501554231282 | 0.0 | 0.0 | 0.0 | True | False |
| topology | gelu | Weather | 192 | tanh_sin001_all | gelu_all | -0.3289329349506762 | 0.0 | 0.0 | 0.3289329349506762 | False | False |
| topology | train_static | Weather | 192 | tanh_sin001_all | gelu_all | -0.3289329349506762 | 0.0 | 0.0 | 0.3289329349506762 | False | False |
| topology | gelu | Weather | 336 | tanh_sin001_all | gelu_all | -0.4212477112537231 | 0.0 | -0.009120909319883413 | 0.4212477112537231 | False | False |
| topology | train_static | Weather | 336 | tanh_sin001_all | gelu_all | -0.4212477112537231 | 0.0 | -0.009120909319883413 | 0.4212477112537231 | False | False |
| topology | gelu | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.04337522953190808 | 0.0 | -0.0068369270057324835 | 0.08771048985107233 | False | False |
| topology | train_static | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.04337522953190808 | 0.0 | -0.0068369270057324835 | 0.08771048985107233 | False | False |
| topology | gelu | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.36685287158734464 | 0.0 | -0.006976507338264325 | 0.4149370328157065 | False | False |
| topology | train_static | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.36685287158734464 | 0.0 | -0.006976507338264325 | 0.4149370328157065 | False | False |
| topology | gelu | Exchange | 336 | tanh_sin001_all | gelu_all | -0.069007027863576 | 0.0 | -0.03619208575301774 | 0.069007027863576 | False | False |
| topology | train_static | Exchange | 336 | tanh_sin001_all | gelu_all | -0.069007027863576 | 0.0 | -0.03619208575301774 | 0.069007027863576 | False | False |
| topology | gelu | ILI | 24 | tanh_sin001_all | enc_gelu_dec_tanh | 0.0019359268347648065 | 0.0 | 0.0 | 0.005604366165226573 | False | False |
| topology | train_static | ILI | 24 | tanh_sin001_all | enc_gelu_dec_tanh | 0.0019359268347648065 | 0.0 | 0.0 | 0.005604366165226573 | False | False |
| topology | gelu | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03254711269524245 | 0.0 | 0.0 | 0.013523091713047096 | False | False |
| topology | train_static | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03254711269524245 | 0.0 | 0.0 | 0.013523091713047096 | False | False |
| topology | gelu | ILI | 48 | tanh_sin001_all | tanh_sin001_all | 0.05369722601898491 | 0.0 | 0.0 | 0.0 | True | False |
| topology | train_static | ILI | 48 | tanh_sin001_all | tanh_sin001_all | 0.05369722601898491 | 0.0 | 0.0 | 0.0 | True | False |

## Interpretation Boundary

- A one-seed oracle matrix is a triage signal only; do not write it as a statistical conclusion.
- Zero-shot rows use only old Informer full-grid rules plus input factors from the new datasets; they do not train on new dataset test errors.
- `new_best_static` is an evaluation oracle for the new matrix, not a deployable zero-shot baseline.
