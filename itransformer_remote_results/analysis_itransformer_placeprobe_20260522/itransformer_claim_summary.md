# iTransformer Activation Transfer Claim Summary

- Raw finite seed-runs: 36
- Complete config cells: 36
- Dataset-horizon cells with non-GELU best activation: 3/3

## Static Ranking

| config_name | mean_rank | best_cells | mean_mse |
| --- | --- | --- | --- |
| swish_ffn_outtanh | 5.333333333333333 | 0 | 0.29843440155188233 |
| tanh_ffn_outtanh | 5.666666666666667 | 1 | 0.29840369025866187 |
| swish_ffn_linear | 5.666666666666667 | 0 | 0.2999190539121627 |
| tanh_sin001_ffn_outtanh | 6.0 | 0 | 0.2984147916237513 |
| gelu_ffn_outtanh | 6.0 | 1 | 0.2985555032889048 |
| gelu_ffn_linear | 6.0 | 0 | 0.29997170964876807 |
| relu_ffn_outtanh | 6.333333333333333 | 0 | 0.2985512415568034 |
| softsign_ffn_outtanh | 6.333333333333333 | 0 | 0.2987219542264938 |
| relu_ffn_linear | 6.666666666666667 | 0 | 0.3001007040341695 |
| tanh_sin001_ffn_linear | 7.333333333333333 | 1 | 0.3003487139940262 |
| tanh_ffn_linear | 8.0 | 0 | 0.3003760427236557 |
| softsign_ffn_linear | 8.666666666666666 | 0 | 0.3007883628209432 |

## Best Cells

| dataset | pred_len | config_name | mse_mean | mae_mean | rank |
| --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | gelu_ffn_outtanh | 0.2996672689914703 | 0.3488405048847198 | 1.0 |
| ETTm2 | 96 | tanh_sin001_ffn_linear | 0.1838691085577011 | 0.2682293057441711 | 1.0 |
| ETTm2 | 720 | tanh_ffn_outtanh | 0.408508151769638 | 0.4024522304534912 | 1.0 |

## Paired vs GELU

| config_name | seed_win_rate | mean_gain_vs_gelu | paired_seed_runs |
| --- | --- | --- | --- |
| swish_ffn_outtanh | 0.6666666666666666 | 0.0035598849217223713 | 3 |
| tanh_ffn_outtanh | 0.3333333333333333 | 0.003182181495285062 | 3 |
| gelu_ffn_outtanh | 0.6666666666666666 | 0.0031628645089605725 | 3 |
| tanh_sin001_ffn_outtanh | 0.3333333333333333 | 0.003153855935585467 | 3 |
| relu_ffn_outtanh | 0.6666666666666666 | 0.0030720053070858217 | 3 |
| softsign_ffn_outtanh | 0.3333333333333333 | 0.0023601410258106506 | 3 |
| swish_ffn_linear | 0.6666666666666666 | 0.00030931273248507505 | 3 |
| gelu_ffn_linear | 0.0 | 0.0 | 3 |
| relu_ffn_linear | 0.3333333333333333 | -0.00037928045592606146 | 3 |
| tanh_sin001_ffn_linear | 0.3333333333333333 | -0.0009048736840607664 | 3 |
| tanh_ffn_linear | 0.3333333333333333 | -0.00097762978938001 | 3 |
| softsign_ffn_linear | 0.3333333333333333 | -0.0022907183098012894 | 3 |
