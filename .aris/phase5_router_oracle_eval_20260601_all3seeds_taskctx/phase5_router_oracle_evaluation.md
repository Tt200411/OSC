# Phase-5 Router Oracle Evaluation

Generated: 2026-05-31T20:36:47.663927+00:00

## Evidence Status

- Expected seed-runs: 189
- Observed deduped seed-runs: 189
- Missing seed-runs: 0
- Aggregated dataset-horizon-config cells: 63
- Dataset-horizon oracle cells: 9

## Oracle Cells

| dataset | pred_len | gelu_mse | new_best_static_config | new_best_static_mse | oracle_config | oracle_mse | oracle_gain_vs_gelu |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Exchange | 96 | 0.9770137333333334 | tanh_sin001_all | 1.0269599666666667 | enc_gelu_dec_tanh | 0.9432436666666666 | 0.03456457725671016 |
| Exchange | 192 | 1.1489570666666669 | softsign_all | 1.4850231999999999 | enc_gelu_dec_tanh | 1.1323604333333332 | 0.014444955181383286 |
| Exchange | 336 | 1.6444349666666664 | softsign_all | 1.7441468333333334 | enc_gelu_dec_tanh | 1.631964 | 0.007583739654931813 |
| ILI | 24 | 5.831972666666666 | tanh_sin001_all | 5.711880533333333 | tanh_sin001_all | 5.711880533333333 | 0.02059202609431526 |
| ILI | 36 | 5.7128608000000005 | tanh_sin001_all | 5.502567933333334 | enc_tanh_dec_gelu | 5.483584533333333 | 0.04013335431990009 |
| ILI | 48 | 5.197631933333334 | tanh_all | 5.020567766666667 | tanh_all | 5.020567766666667 | 0.0340663149945503 |
| Weather | 96 | 0.36682802 | tanh_sin001_all | 0.34896052333333333 | tanh_sin001_all | 0.34896052333333333 | 0.04870810214188832 |
| Weather | 192 | 0.40266667000000006 | tanh_sin001_all | 0.5043701166666666 | enc_gelu_dec_tanh | 0.40160789 | 0.0026294205080346898 |
| Weather | 336 | 0.46741879 | tanh_sin001_all | 0.6646860033333333 | enc_gelu_dec_tanh | 0.4610632733333333 | 0.013597050017323163 |

## Zero-Shot Router Summary

| feature_set | fallback_policy | cell_count | mean_gain_vs_gelu | median_gain_vs_gelu | mean_gain_vs_train_static | mean_gain_vs_new_static | positive_vs_gelu_rate | positive_vs_train_static_rate | positive_vs_new_static_rate | mean_regret_vs_oracle | median_regret_vs_oracle | mean_oracle_capture_ratio | config_hit_rate | action_hit_rate | abstain_rate | top_chosen_configs | top_oracle_configs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_input | gelu | 9 | -0.11119379614786122 | -0.05112132166548862 | 0.0 | -0.005911945505484934 | 0.4444444444444444 | 0.0 | 0.0 | 0.136967430671699 | 0.0887536306454572 | -17.65869386367846 | 0.2222222222222222 | 1.0 | 0.0 | tanh_sin001_all | enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu;tanh_all |
| all_input | train_static | 9 | -0.11119379614786122 | -0.05112132166548862 | 0.0 | -0.005911945505484934 | 0.4444444444444444 | 0.0 | 0.0 | 0.136967430671699 | 0.0887536306454572 | -17.65869386367846 | 0.2222222222222222 | 1.0 | 0.0 | tanh_sin001_all | enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu;tanh_all |
| all_with_topology | gelu | 9 | -0.11119379614786122 | -0.05112132166548862 | 0.0 | -0.005911945505484934 | 0.4444444444444444 | 0.0 | 0.0 | 0.136967430671699 | 0.0887536306454572 | -17.65869386367846 | 0.2222222222222222 | 1.0 | 0.0 | tanh_sin001_all | enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu;tanh_all |
| all_with_topology | train_static | 9 | -0.11119379614786122 | -0.05112132166548862 | 0.0 | -0.005911945505484934 | 0.4444444444444444 | 0.0 | 0.0 | 0.136967430671699 | 0.0887536306454572 | -17.65869386367846 | 0.2222222222222222 | 1.0 | 0.0 | tanh_sin001_all | enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu;tanh_all |
| data | gelu | 9 | -0.11119379614786122 | -0.05112132166548862 | 0.0 | -0.005911945505484934 | 0.4444444444444444 | 0.0 | 0.0 | 0.136967430671699 | 0.0887536306454572 | -17.65869386367846 | 0.2222222222222222 | 1.0 | 0.0 | tanh_sin001_all | enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu;tanh_all |
| data | train_static | 9 | -0.11119379614786122 | -0.05112132166548862 | 0.0 | -0.005911945505484934 | 0.4444444444444444 | 0.0 | 0.0 | 0.136967430671699 | 0.0887536306454572 | -17.65869386367846 | 0.2222222222222222 | 1.0 | 0.0 | tanh_sin001_all | enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu;tanh_all |
| statistical | gelu | 9 | -0.11119379614786122 | -0.05112132166548862 | 0.0 | -0.005911945505484934 | 0.4444444444444444 | 0.0 | 0.0 | 0.136967430671699 | 0.0887536306454572 | -17.65869386367846 | 0.2222222222222222 | 1.0 | 0.0 | tanh_sin001_all | enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu;tanh_all |
| statistical | train_static | 9 | -0.11119379614786122 | -0.05112132166548862 | 0.0 | -0.005911945505484934 | 0.4444444444444444 | 0.0 | 0.0 | 0.136967430671699 | 0.0887536306454572 | -17.65869386367846 | 0.2222222222222222 | 1.0 | 0.0 | tanh_sin001_all | enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu;tanh_all |
| topology | gelu | 9 | -0.11119379614786122 | -0.05112132166548862 | 0.0 | -0.005911945505484934 | 0.4444444444444444 | 0.0 | 0.0 | 0.136967430671699 | 0.0887536306454572 | -17.65869386367846 | 0.2222222222222222 | 1.0 | 0.0 | tanh_sin001_all | enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu;tanh_all |
| topology | train_static | 9 | -0.11119379614786122 | -0.05112132166548862 | 0.0 | -0.005911945505484934 | 0.4444444444444444 | 0.0 | 0.0 | 0.136967430671699 | 0.0887536306454572 | -17.65869386367846 | 0.2222222222222222 | 1.0 | 0.0 | tanh_sin001_all | enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu;tanh_all |
| frequency | gelu | 9 | -0.11162501651745926 | -0.051395353978647584 | -0.0004190592964157505 | -0.006335303932146605 | 0.4444444444444444 | 0.1111111111111111 | 0.0 | 0.13740745462662582 | 0.08903747388002622 | -17.692533664756372 | 0.2222222222222222 | 1.0 | 0.0 | tanh_all;tanh_sin001_all | enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu;tanh_all |
| frequency | train_static | 9 | -0.11162501651745926 | -0.051395353978647584 | -0.0004190592964157505 | -0.006335303932146605 | 0.4444444444444444 | 0.1111111111111111 | 0.0 | 0.13740745462662582 | 0.08903747388002622 | -17.692533664756372 | 0.2222222222222222 | 1.0 | 0.0 | tanh_all;tanh_sin001_all | enc_gelu_dec_tanh;tanh_sin001_all;enc_tanh_dec_gelu;tanh_all |

## By Dataset

| feature_set | fallback_policy | dataset | cell_count | mean_gain_vs_gelu | mean_gain_vs_train_static | mean_gain_vs_new_static | mean_regret_vs_oracle | config_hit_rate | action_hit_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_input | gelu | Exchange | 3 | -0.1549648801833595 | 0.0 | -0.017247069308653926 | 0.1767548077357127 | 0.0 | 1.0 |
| all_input | gelu | ILI | 3 | 0.030017473891370883 | 0.0 | -0.0004887672078008777 | 0.0016427204224427614 | 0.3333333333333333 | 1.0 |
| all_input | gelu | Weather | 3 | -0.208633982151595 | 0.0 | 0.0 | 0.2325047638569416 | 0.3333333333333333 | 1.0 |
| all_input | train_static | Exchange | 3 | -0.1549648801833595 | 0.0 | -0.017247069308653926 | 0.1767548077357127 | 0.0 | 1.0 |
| all_input | train_static | ILI | 3 | 0.030017473891370883 | 0.0 | -0.0004887672078008777 | 0.0016427204224427614 | 0.3333333333333333 | 1.0 |
| all_input | train_static | Weather | 3 | -0.208633982151595 | 0.0 | 0.0 | 0.2325047638569416 | 0.3333333333333333 | 1.0 |
| all_with_topology | gelu | Exchange | 3 | -0.1549648801833595 | 0.0 | -0.017247069308653926 | 0.1767548077357127 | 0.0 | 1.0 |
| all_with_topology | gelu | ILI | 3 | 0.030017473891370883 | 0.0 | -0.0004887672078008777 | 0.0016427204224427614 | 0.3333333333333333 | 1.0 |
| all_with_topology | gelu | Weather | 3 | -0.208633982151595 | 0.0 | 0.0 | 0.2325047638569416 | 0.3333333333333333 | 1.0 |
| all_with_topology | train_static | Exchange | 3 | -0.1549648801833595 | 0.0 | -0.017247069308653926 | 0.1767548077357127 | 0.0 | 1.0 |
| all_with_topology | train_static | ILI | 3 | 0.030017473891370883 | 0.0 | -0.0004887672078008777 | 0.0016427204224427614 | 0.3333333333333333 | 1.0 |
| all_with_topology | train_static | Weather | 3 | -0.208633982151595 | 0.0 | 0.0 | 0.2325047638569416 | 0.3333333333333333 | 1.0 |
| data | gelu | Exchange | 3 | -0.1549648801833595 | 0.0 | -0.017247069308653926 | 0.1767548077357127 | 0.0 | 1.0 |
| data | gelu | ILI | 3 | 0.030017473891370883 | 0.0 | -0.0004887672078008777 | 0.0016427204224427614 | 0.3333333333333333 | 1.0 |
| data | gelu | Weather | 3 | -0.208633982151595 | 0.0 | 0.0 | 0.2325047638569416 | 0.3333333333333333 | 1.0 |
| data | train_static | Exchange | 3 | -0.1549648801833595 | 0.0 | -0.017247069308653926 | 0.1767548077357127 | 0.0 | 1.0 |
| data | train_static | ILI | 3 | 0.030017473891370883 | 0.0 | -0.0004887672078008777 | 0.0016427204224427614 | 0.3333333333333333 | 1.0 |
| data | train_static | Weather | 3 | -0.208633982151595 | 0.0 | 0.0 | 0.2325047638569416 | 0.3333333333333333 | 1.0 |
| frequency | gelu | Exchange | 3 | -0.15561324323631262 | -0.0005944982886029773 | -0.017855180618814293 | 0.17741086210513726 | 0.0 | 1.0 |
| frequency | gelu | ILI | 3 | 0.02937217583552981 | -0.0006626796006442744 | -0.0011507311776255203 | 0.002306737917798593 | 0.3333333333333333 | 1.0 |
| frequency | gelu | Weather | 3 | -0.208633982151595 | 0.0 | 0.0 | 0.2325047638569416 | 0.3333333333333333 | 1.0 |
| frequency | train_static | Exchange | 3 | -0.15561324323631262 | -0.0005944982886029773 | -0.017855180618814293 | 0.17741086210513726 | 0.0 | 1.0 |
| frequency | train_static | ILI | 3 | 0.02937217583552981 | -0.0006626796006442744 | -0.0011507311776255203 | 0.002306737917798593 | 0.3333333333333333 | 1.0 |
| frequency | train_static | Weather | 3 | -0.208633982151595 | 0.0 | 0.0 | 0.2325047638569416 | 0.3333333333333333 | 1.0 |
| statistical | gelu | Exchange | 3 | -0.1549648801833595 | 0.0 | -0.017247069308653926 | 0.1767548077357127 | 0.0 | 1.0 |
| statistical | gelu | ILI | 3 | 0.030017473891370883 | 0.0 | -0.0004887672078008777 | 0.0016427204224427614 | 0.3333333333333333 | 1.0 |
| statistical | gelu | Weather | 3 | -0.208633982151595 | 0.0 | 0.0 | 0.2325047638569416 | 0.3333333333333333 | 1.0 |
| statistical | train_static | Exchange | 3 | -0.1549648801833595 | 0.0 | -0.017247069308653926 | 0.1767548077357127 | 0.0 | 1.0 |
| statistical | train_static | ILI | 3 | 0.030017473891370883 | 0.0 | -0.0004887672078008777 | 0.0016427204224427614 | 0.3333333333333333 | 1.0 |
| statistical | train_static | Weather | 3 | -0.208633982151595 | 0.0 | 0.0 | 0.2325047638569416 | 0.3333333333333333 | 1.0 |
| topology | gelu | Exchange | 3 | -0.1549648801833595 | 0.0 | -0.017247069308653926 | 0.1767548077357127 | 0.0 | 1.0 |
| topology | gelu | ILI | 3 | 0.030017473891370883 | 0.0 | -0.0004887672078008777 | 0.0016427204224427614 | 0.3333333333333333 | 1.0 |
| topology | gelu | Weather | 3 | -0.208633982151595 | 0.0 | 0.0 | 0.2325047638569416 | 0.3333333333333333 | 1.0 |
| topology | train_static | Exchange | 3 | -0.1549648801833595 | 0.0 | -0.017247069308653926 | 0.1767548077357127 | 0.0 | 1.0 |
| topology | train_static | ILI | 3 | 0.030017473891370883 | 0.0 | -0.0004887672078008777 | 0.0016427204224427614 | 0.3333333333333333 | 1.0 |
| topology | train_static | Weather | 3 | -0.208633982151595 | 0.0 | 0.0 | 0.2325047638569416 | 0.3333333333333333 | 1.0 |

## Cell-Level Router Choices

| feature_set | fallback_policy | dataset | pred_len | chosen_config | oracle_config | gain_vs_gelu | gain_vs_train_static | gain_vs_new_static | regret_vs_oracle | config_hit | abstained |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_input | gelu | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.04870810214188832 | 0.0 | 0.0 | 0.0 | True | False |
| all_input | train_static | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.04870810214188832 | 0.0 | 0.0 | 0.0 | True | False |
| all_input | gelu | Weather | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.25257478267736083 | 0.0 | 0.0 | 0.25587701144682845 | False | False |
| all_input | train_static | Weather | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.25257478267736083 | 0.0 | 0.0 | 0.25587701144682845 | False | False |
| all_input | gelu | Weather | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.42203526591931256 | 0.0 | 0.0 | 0.4416372801239963 | False | False |
| all_input | train_static | Weather | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.42203526591931256 | 0.0 | 0.0 | 0.4416372801239963 | False | False |
| all_input | gelu | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.05112132166548862 | 0.0 | 0.0 | 0.0887536306454572 | False | False |
| all_input | train_static | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.05112132166548862 | 0.0 | 0.0 | 0.0887536306454572 | False | False |
| all_input | gelu | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.32461726449192496 | 0.0 | -0.02485157583172213 | 0.34403174277872006 | False | False |
| all_input | train_static | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.32461726449192496 | 0.0 | -0.02485157583172213 | 0.34403174277872006 | False | False |
| all_input | gelu | Exchange | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.08915605439266491 | 0.0 | -0.026889632094239644 | 0.09747904978296083 | False | False |
| all_input | train_static | Exchange | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.08915605439266491 | 0.0 | -0.026889632094239644 | 0.09747904978296083 | False | False |
| all_input | gelu | ILI | 24 | tanh_sin001_all | tanh_sin001_all | 0.02059202609431526 | 0.0 | 0.0 | 0.0 | True | False |
| all_input | train_static | ILI | 24 | tanh_sin001_all | tanh_sin001_all | 0.02059202609431526 | 0.0 | 0.0 | 0.0 | True | False |
| all_input | gelu | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03681043071566987 | 0.0 | 0.0 | 0.003461859643925651 | False | False |
| all_input | train_static | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03681043071566987 | 0.0 | 0.0 | 0.003461859643925651 | False | False |
| all_input | gelu | ILI | 48 | tanh_sin001_all | tanh_all | 0.03264996486412752 | 0.0 | -0.0014663016234026333 | 0.0014663016234026333 | False | False |
| all_input | train_static | ILI | 48 | tanh_sin001_all | tanh_all | 0.03264996486412752 | 0.0 | -0.0014663016234026333 | 0.0014663016234026333 | False | False |
| all_with_topology | gelu | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.04870810214188832 | 0.0 | 0.0 | 0.0 | True | False |
| all_with_topology | train_static | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.04870810214188832 | 0.0 | 0.0 | 0.0 | True | False |
| all_with_topology | gelu | Weather | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.25257478267736083 | 0.0 | 0.0 | 0.25587701144682845 | False | False |
| all_with_topology | train_static | Weather | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.25257478267736083 | 0.0 | 0.0 | 0.25587701144682845 | False | False |
| all_with_topology | gelu | Weather | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.42203526591931256 | 0.0 | 0.0 | 0.4416372801239963 | False | False |
| all_with_topology | train_static | Weather | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.42203526591931256 | 0.0 | 0.0 | 0.4416372801239963 | False | False |
| all_with_topology | gelu | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.05112132166548862 | 0.0 | 0.0 | 0.0887536306454572 | False | False |
| all_with_topology | train_static | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.05112132166548862 | 0.0 | 0.0 | 0.0887536306454572 | False | False |
| all_with_topology | gelu | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.32461726449192496 | 0.0 | -0.02485157583172213 | 0.34403174277872006 | False | False |
| all_with_topology | train_static | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.32461726449192496 | 0.0 | -0.02485157583172213 | 0.34403174277872006 | False | False |
| all_with_topology | gelu | Exchange | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.08915605439266491 | 0.0 | -0.026889632094239644 | 0.09747904978296083 | False | False |
| all_with_topology | train_static | Exchange | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.08915605439266491 | 0.0 | -0.026889632094239644 | 0.09747904978296083 | False | False |
| all_with_topology | gelu | ILI | 24 | tanh_sin001_all | tanh_sin001_all | 0.02059202609431526 | 0.0 | 0.0 | 0.0 | True | False |
| all_with_topology | train_static | ILI | 24 | tanh_sin001_all | tanh_sin001_all | 0.02059202609431526 | 0.0 | 0.0 | 0.0 | True | False |
| all_with_topology | gelu | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03681043071566987 | 0.0 | 0.0 | 0.003461859643925651 | False | False |
| all_with_topology | train_static | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03681043071566987 | 0.0 | 0.0 | 0.003461859643925651 | False | False |
| all_with_topology | gelu | ILI | 48 | tanh_sin001_all | tanh_all | 0.03264996486412752 | 0.0 | -0.0014663016234026333 | 0.0014663016234026333 | False | False |
| all_with_topology | train_static | ILI | 48 | tanh_sin001_all | tanh_all | 0.03264996486412752 | 0.0 | -0.0014663016234026333 | 0.0014663016234026333 | False | False |
| data | gelu | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.04870810214188832 | 0.0 | 0.0 | 0.0 | True | False |
| data | train_static | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.04870810214188832 | 0.0 | 0.0 | 0.0 | True | False |
| data | gelu | Weather | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.25257478267736083 | 0.0 | 0.0 | 0.25587701144682845 | False | False |
| data | train_static | Weather | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.25257478267736083 | 0.0 | 0.0 | 0.25587701144682845 | False | False |
| data | gelu | Weather | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.42203526591931256 | 0.0 | 0.0 | 0.4416372801239963 | False | False |
| data | train_static | Weather | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.42203526591931256 | 0.0 | 0.0 | 0.4416372801239963 | False | False |
| data | gelu | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.05112132166548862 | 0.0 | 0.0 | 0.0887536306454572 | False | False |
| data | train_static | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.05112132166548862 | 0.0 | 0.0 | 0.0887536306454572 | False | False |
| data | gelu | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.32461726449192496 | 0.0 | -0.02485157583172213 | 0.34403174277872006 | False | False |
| data | train_static | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.32461726449192496 | 0.0 | -0.02485157583172213 | 0.34403174277872006 | False | False |
| data | gelu | Exchange | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.08915605439266491 | 0.0 | -0.026889632094239644 | 0.09747904978296083 | False | False |
| data | train_static | Exchange | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.08915605439266491 | 0.0 | -0.026889632094239644 | 0.09747904978296083 | False | False |
| data | gelu | ILI | 24 | tanh_sin001_all | tanh_sin001_all | 0.02059202609431526 | 0.0 | 0.0 | 0.0 | True | False |
| data | train_static | ILI | 24 | tanh_sin001_all | tanh_sin001_all | 0.02059202609431526 | 0.0 | 0.0 | 0.0 | True | False |
| data | gelu | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03681043071566987 | 0.0 | 0.0 | 0.003461859643925651 | False | False |
| data | train_static | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03681043071566987 | 0.0 | 0.0 | 0.003461859643925651 | False | False |
| data | gelu | ILI | 48 | tanh_sin001_all | tanh_all | 0.03264996486412752 | 0.0 | -0.0014663016234026333 | 0.0014663016234026333 | False | False |
| data | train_static | ILI | 48 | tanh_sin001_all | tanh_all | 0.03264996486412752 | 0.0 | -0.0014663016234026333 | 0.0014663016234026333 | False | False |
| frequency | gelu | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.04870810214188832 | 0.0 | 0.0 | 0.0 | True | False |
| frequency | train_static | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.04870810214188832 | 0.0 | 0.0 | 0.0 | True | False |
| frequency | gelu | Weather | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.25257478267736083 | 0.0 | 0.0 | 0.25587701144682845 | False | False |
| frequency | train_static | Weather | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.25257478267736083 | 0.0 | 0.0 | 0.25587701144682845 | False | False |
| frequency | gelu | Weather | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.42203526591931256 | 0.0 | 0.0 | 0.4416372801239963 | False | False |
| frequency | train_static | Weather | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.42203526591931256 | 0.0 | 0.0 | 0.4416372801239963 | False | False |
| frequency | gelu | Exchange | 96 | tanh_all | enc_gelu_dec_tanh | -0.051395353978647584 | -0.0002607047421745411 | -0.0002607047421745411 | 0.08903747388002622 | False | False |
| frequency | train_static | Exchange | 96 | tanh_all | enc_gelu_dec_tanh | -0.051395353978647584 | -0.0002607047421745411 | -0.0002607047421745411 | 0.08903747388002622 | False | False |
| frequency | gelu | Exchange | 192 | tanh_all | enc_gelu_dec_tanh | -0.32468758913881063 | -5.309054077037245e-05 | -0.024905985756092406 | 0.34410309815075674 | False | False |
| frequency | train_static | Exchange | 192 | tanh_all | enc_gelu_dec_tanh | -0.32468758913881063 | -5.309054077037245e-05 | -0.024905985756092406 | 0.34410309815075674 | False | False |
| frequency | gelu | Exchange | 336 | tanh_all | enc_gelu_dec_tanh | -0.09075678659147964 | -0.0014696995828640182 | -0.028398851358175933 | 0.09909201428462885 | False | False |
| frequency | train_static | Exchange | 336 | tanh_all | enc_gelu_dec_tanh | -0.09075678659147964 | -0.0014696995828640182 | -0.028398851358175933 | 0.09909201428462885 | False | False |
| frequency | gelu | ILI | 24 | tanh_all | tanh_sin001_all | 0.018953832545397127 | -0.0016726365238638441 | -0.0016726365238638441 | 0.0016726365238638441 | False | False |
| frequency | train_static | ILI | 24 | tanh_all | tanh_sin001_all | 0.018953832545397127 | -0.0016726365238638441 | -0.0016726365238638441 | 0.0016726365238638441 | False | False |
| frequency | gelu | ILI | 36 | tanh_all | enc_tanh_dec_gelu | 0.035096379966642 | -0.0017795570090127168 | -0.0017795570090127168 | 0.005247577229531934 | False | False |
| frequency | train_static | ILI | 36 | tanh_all | enc_tanh_dec_gelu | 0.035096379966642 | -0.0017795570090127168 | -0.0017795570090127168 | 0.005247577229531934 | False | False |
| frequency | gelu | ILI | 48 | tanh_all | tanh_all | 0.0340663149945503 | 0.0014641547309437379 | 0.0 | 0.0 | True | False |
| frequency | train_static | ILI | 48 | tanh_all | tanh_all | 0.0340663149945503 | 0.0014641547309437379 | 0.0 | 0.0 | True | False |
| statistical | gelu | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.04870810214188832 | 0.0 | 0.0 | 0.0 | True | False |
| statistical | train_static | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.04870810214188832 | 0.0 | 0.0 | 0.0 | True | False |
| statistical | gelu | Weather | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.25257478267736083 | 0.0 | 0.0 | 0.25587701144682845 | False | False |
| statistical | train_static | Weather | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.25257478267736083 | 0.0 | 0.0 | 0.25587701144682845 | False | False |
| statistical | gelu | Weather | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.42203526591931256 | 0.0 | 0.0 | 0.4416372801239963 | False | False |
| statistical | train_static | Weather | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.42203526591931256 | 0.0 | 0.0 | 0.4416372801239963 | False | False |
| statistical | gelu | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.05112132166548862 | 0.0 | 0.0 | 0.0887536306454572 | False | False |
| statistical | train_static | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.05112132166548862 | 0.0 | 0.0 | 0.0887536306454572 | False | False |
| statistical | gelu | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.32461726449192496 | 0.0 | -0.02485157583172213 | 0.34403174277872006 | False | False |
| statistical | train_static | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.32461726449192496 | 0.0 | -0.02485157583172213 | 0.34403174277872006 | False | False |
| statistical | gelu | Exchange | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.08915605439266491 | 0.0 | -0.026889632094239644 | 0.09747904978296083 | False | False |
| statistical | train_static | Exchange | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.08915605439266491 | 0.0 | -0.026889632094239644 | 0.09747904978296083 | False | False |
| statistical | gelu | ILI | 24 | tanh_sin001_all | tanh_sin001_all | 0.02059202609431526 | 0.0 | 0.0 | 0.0 | True | False |
| statistical | train_static | ILI | 24 | tanh_sin001_all | tanh_sin001_all | 0.02059202609431526 | 0.0 | 0.0 | 0.0 | True | False |
| statistical | gelu | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03681043071566987 | 0.0 | 0.0 | 0.003461859643925651 | False | False |
| statistical | train_static | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03681043071566987 | 0.0 | 0.0 | 0.003461859643925651 | False | False |
| statistical | gelu | ILI | 48 | tanh_sin001_all | tanh_all | 0.03264996486412752 | 0.0 | -0.0014663016234026333 | 0.0014663016234026333 | False | False |
| statistical | train_static | ILI | 48 | tanh_sin001_all | tanh_all | 0.03264996486412752 | 0.0 | -0.0014663016234026333 | 0.0014663016234026333 | False | False |
| topology | gelu | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.04870810214188832 | 0.0 | 0.0 | 0.0 | True | False |
| topology | train_static | Weather | 96 | tanh_sin001_all | tanh_sin001_all | 0.04870810214188832 | 0.0 | 0.0 | 0.0 | True | False |
| topology | gelu | Weather | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.25257478267736083 | 0.0 | 0.0 | 0.25587701144682845 | False | False |
| topology | train_static | Weather | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.25257478267736083 | 0.0 | 0.0 | 0.25587701144682845 | False | False |
| topology | gelu | Weather | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.42203526591931256 | 0.0 | 0.0 | 0.4416372801239963 | False | False |
| topology | train_static | Weather | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.42203526591931256 | 0.0 | 0.0 | 0.4416372801239963 | False | False |
| topology | gelu | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.05112132166548862 | 0.0 | 0.0 | 0.0887536306454572 | False | False |
| topology | train_static | Exchange | 96 | tanh_sin001_all | enc_gelu_dec_tanh | -0.05112132166548862 | 0.0 | 0.0 | 0.0887536306454572 | False | False |
| topology | gelu | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.32461726449192496 | 0.0 | -0.02485157583172213 | 0.34403174277872006 | False | False |
| topology | train_static | Exchange | 192 | tanh_sin001_all | enc_gelu_dec_tanh | -0.32461726449192496 | 0.0 | -0.02485157583172213 | 0.34403174277872006 | False | False |
| topology | gelu | Exchange | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.08915605439266491 | 0.0 | -0.026889632094239644 | 0.09747904978296083 | False | False |
| topology | train_static | Exchange | 336 | tanh_sin001_all | enc_gelu_dec_tanh | -0.08915605439266491 | 0.0 | -0.026889632094239644 | 0.09747904978296083 | False | False |
| topology | gelu | ILI | 24 | tanh_sin001_all | tanh_sin001_all | 0.02059202609431526 | 0.0 | 0.0 | 0.0 | True | False |
| topology | train_static | ILI | 24 | tanh_sin001_all | tanh_sin001_all | 0.02059202609431526 | 0.0 | 0.0 | 0.0 | True | False |
| topology | gelu | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03681043071566987 | 0.0 | 0.0 | 0.003461859643925651 | False | False |
| topology | train_static | ILI | 36 | tanh_sin001_all | enc_tanh_dec_gelu | 0.03681043071566987 | 0.0 | 0.0 | 0.003461859643925651 | False | False |
| topology | gelu | ILI | 48 | tanh_sin001_all | tanh_all | 0.03264996486412752 | 0.0 | -0.0014663016234026333 | 0.0014663016234026333 | False | False |
| topology | train_static | ILI | 48 | tanh_sin001_all | tanh_all | 0.03264996486412752 | 0.0 | -0.0014663016234026333 | 0.0014663016234026333 | False | False |

## Interpretation Boundary

- A one-seed oracle matrix is a triage signal only; do not write it as a statistical conclusion.
- Zero-shot rows use only old Informer full-grid rules plus input factors from the new datasets; they do not train on new dataset test errors.
- `new_best_static` is an evaluation oracle for the new matrix, not a deployable zero-shot baseline.
