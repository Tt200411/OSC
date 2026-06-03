# PatchTST Transfer Report

- Raw finite seed-runs: 198
- Stage1 dataset-horizon cells: 12
- Stage2 dataset-horizon cells: 6
- Complete config cells: 66
- Best cells using output tanh placement: 0/12

## Overall Ranking

| config_name | mean_rank | best_cells | mean_mse |
| --- | --- | --- | --- |
| gelu_ffn_linear | 2.25 | 7 | 0.250040748467048 |
| tanh_ffn_linear | 2.5 | 0 | 0.25147366130517584 |
| tanh_sin001_ffn_linear | 2.5833333333333335 | 2 | 0.2514677879710992 |
| softsign_ffn_linear | 2.9166666666666665 | 3 | 0.25178682721323437 |
| gelu_ffn_outtanh | 5.5 | 0 | 0.3405648014611668 |
| tanh_ffn_outtanh | 5.5 | 0 | 0.34168465849426055 |
| tanh_sin001_ffn_outtanh | 6.5 | 0 | 0.34183781759606463 |

## Output Tanh vs Linear Placement

| base_activation | paired_seed_runs | output_tanh_win_rate | output_tanh_gain_mean | output_tanh_gain_median |
| --- | --- | --- | --- | --- |
| tanh | 18 | 0.0 | -0.052759788684054655 | -0.020540342626063038 |
| tanh_sin001 | 18 | 0.0 | -0.05313216199814223 | -0.020565487125148704 |
| gelu | 18 | 0.0 | -0.054397149426369314 | -0.02307109079686677 |

## Paired Gain vs GELU Linear

| config_name | seed_win_rate | mean_gain_vs_gelu_linear | median_gain_vs_gelu_linear | paired_seed_runs |
| --- | --- | --- | --- | --- |
| gelu_ffn_linear | 0.0 | 0.0 | 0.0 | 36 |
| tanh_sin001_ffn_linear | 0.3888888888888889 | -0.0028084364177616266 | -0.006933476818636642 | 36 |
| tanh_ffn_linear | 0.3888888888888889 | -0.0028380871422807427 | -0.007105577061741556 | 36 |
| softsign_ffn_linear | 0.3888888888888889 | -0.0037343864147031647 | -0.00687143327207613 | 36 |
| gelu_ffn_outtanh | 0.0 | -0.054397149426369314 | -0.02307109079686677 | 18 |
| tanh_ffn_outtanh | 0.05555555555555555 | -0.056893942595955455 | -0.028237572284196412 | 18 |
| tanh_sin001_ffn_outtanh | 0.05555555555555555 | -0.057295434766353584 | -0.028327568743888767 | 18 |

## Best Cells

| dataset | pred_len | config_name | mse_mean | mse_std | mae_mean | rank_in_cell |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 168 | gelu_ffn_linear | 0.32587979237238557 | 0.0017459829345106363 | 0.37059006094932556 | 1.0 |
| ETTh2 | 336 | gelu_ffn_linear | 0.3284500340620677 | 0.0005283864522811148 | 0.38438204924265545 | 1.0 |
| ETTh2 | 720 | softsign_ffn_linear | 0.37831496198972064 | 0.0013728190625910326 | 0.41821983456611633 | 1.0 |
| ETTm2 | 96 | gelu_ffn_linear | 0.1655888458093007 | 0.0009037757389714093 | 0.2543961803118388 | 1.0 |
| ETTm2 | 288 | gelu_ffn_linear | 0.26273878415425617 | 0.0020405863438131035 | 0.3208762804667155 | 1.0 |
| ETTm2 | 672 | gelu_ffn_linear | 0.35523249705632526 | 0.0007543683803074536 | 0.3773835599422455 | 1.0 |
| Solar2 | 24 | tanh_sin001_ffn_linear | 0.06517332047224039 | 0.0005467123560755935 | 0.130756268898646 | 1.0 |
| Solar2 | 96 | softsign_ffn_linear | 0.09855036189158754 | 0.0003018434408274542 | 0.1776194224754969 | 1.0 |
| Solar3 | 24 | gelu_ffn_linear | 0.2790345549583435 | 0.0023341723919720267 | 0.3030668795108795 | 1.0 |
| Solar3 | 96 | gelu_ffn_linear | 0.4464092453320821 | 0.0012883108688151407 | 0.418380469083786 | 1.0 |
| Solar5 | 24 | tanh_sin001_ffn_linear | 0.1132596855362256 | 0.00018478889116685021 | 0.16209468245506278 | 1.0 |
| Solar5 | 96 | softsign_ffn_linear | 0.17458280920982358 | 0.0019145211080910645 | 0.22406789660453794 | 1.0 |
