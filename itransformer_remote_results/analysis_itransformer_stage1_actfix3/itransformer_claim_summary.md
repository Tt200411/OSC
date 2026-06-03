# iTransformer Activation Transfer Claim Summary

- Raw finite seed-runs: 24
- Complete config cells: 24
- Dataset-horizon cells with non-GELU best activation: 1/6

## Static Ranking

| config_name | mean_rank | best_cells | mean_mse |
| --- | --- | --- | --- |
| gelu_ffn_linear | 1.5 | 5 | 0.34420233964920044 |
| softsign_ffn_linear | 2.8333333333333335 | 0 | 0.34665238360563916 |
| tanh_ffn_linear | 2.8333333333333335 | 0 | 0.34709813445806503 |
| tanh_sin001_ffn_linear | 2.8333333333333335 | 1 | 0.34710247069597244 |

## Best Cells

| dataset | pred_len | config_name | mse_mean | mae_mean | rank |
| --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | gelu_ffn_linear | 0.3015976250171661 | 0.3514589667320251 | 1.0 |
| ETTh2 | 336 | gelu_ffn_linear | 0.4228940308094024 | 0.4321351647377014 | 1.0 |
| ETTh2 | 720 | gelu_ffn_linear | 0.4289864897727966 | 0.4472371935844421 | 1.0 |
| ETTm2 | 96 | tanh_sin001_ffn_linear | 0.1838691085577011 | 0.2682293057441711 | 1.0 |
| ETTm2 | 336 | gelu_ffn_linear | 0.3134183883666992 | 0.3513455986976623 | 1.0 |
| ETTm2 | 720 | gelu_ffn_linear | 0.414091557264328 | 0.4065375030040741 | 1.0 |

## Paired vs GELU

| config_name | seed_win_rate | mean_gain_vs_gelu | paired_seed_runs |
| --- | --- | --- | --- |
| gelu_ffn_linear | 0.0 | 0.0 | 6 |
| softsign_ffn_linear | 0.16666666666666666 | -0.006254051431291115 | 6 |
| tanh_ffn_linear | 0.16666666666666666 | -0.007112938095173896 | 6 |
| tanh_sin001_ffn_linear | 0.16666666666666666 | -0.007118417783886638 | 6 |
