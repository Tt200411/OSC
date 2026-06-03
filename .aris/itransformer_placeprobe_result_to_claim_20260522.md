# iTransformer Regime/Placement Probe Summary

**Date**: 2026-05-22

## Scope

Small positive-signal matrix on cloud RTX 4090:

- Cells: ETTh2/96, ETTm2/96, ETTm2/720
- FFN regimes: GELU, ReLU, Swish, tanh, softsign, tanh_sin001
- Placements: FFN + linear output, FFN + output tanh
- Seed: 2024
- Expected rows: 36
- Reused rows: 12 existing linear rows
- Newly launched rows: 24
- Completed rows: 36/36 finite after aggregation

## Best vs Default GELU Linear

| dataset | pred_len | config_name | mse | gelu_linear_mse | gain_vs_gelu_linear_pct |
| --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | gelu_ffn_outtanh | 0.299667 | 0.301598 | 0.640 |
| ETTm2 | 96 | tanh_sin001_ffn_linear | 0.183869 | 0.184226 | 0.194 |
| ETTm2 | 720 | tanh_ffn_outtanh | 0.408508 | 0.414092 | 1.348 |

## Static Ranking Across Probe Cells

| config_name | mean_rank | best_cells | mean_mse |
| --- | --- | --- | --- |
| swish_ffn_outtanh | 5.33 | 0 | 0.298434 |
| tanh_ffn_outtanh | 5.67 | 1 | 0.298404 |
| swish_ffn_linear | 5.67 | 0 | 0.299919 |
| tanh_sin001_ffn_outtanh | 6.00 | 0 | 0.298415 |
| gelu_ffn_outtanh | 6.00 | 1 | 0.298556 |
| gelu_ffn_linear | 6.00 | 0 | 0.299972 |
| relu_ffn_outtanh | 6.33 | 0 | 0.298551 |
| softsign_ffn_outtanh | 6.33 | 0 | 0.298722 |
| relu_ffn_linear | 6.67 | 0 | 0.300101 |
| tanh_sin001_ffn_linear | 7.33 | 1 | 0.300349 |
| tanh_ffn_linear | 8.00 | 0 | 0.300376 |
| softsign_ffn_linear | 8.67 | 0 | 0.300788 |

## Output-Tanh Placement Pairs

| dataset | pred_len | activation_base | linear_mse | outtanh_mse | outtanh_gain_pct | outtanh_win |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | gelu | 0.301598 | 0.299667 | 0.640 | True |
| ETTh2 | 96 | relu | 0.301528 | 0.299734 | 0.595 | True |
| ETTh2 | 96 | softsign | 0.302702 | 0.301622 | 0.357 | True |
| ETTh2 | 96 | swish | 0.301681 | 0.300073 | 0.533 | True |
| ETTh2 | 96 | tanh | 0.302767 | 0.301993 | 0.255 | True |
| ETTh2 | 96 | tanh_sin001 | 0.302773 | 0.302011 | 0.252 | True |
| ETTm2 | 96 | gelu | 0.184226 | 0.185059 | -0.452 | False |
| ETTm2 | 96 | relu | 0.184314 | 0.185140 | -0.448 | False |
| ETTm2 | 96 | softsign | 0.184213 | 0.184874 | -0.359 | False |
| ETTm2 | 96 | swish | 0.184020 | 0.184834 | -0.442 | False |
| ETTm2 | 96 | tanh | 0.183878 | 0.184709 | -0.452 | False |
| ETTm2 | 96 | tanh_sin001 | 0.183869 | 0.184706 | -0.455 | False |
| ETTm2 | 720 | gelu | 0.414092 | 0.410940 | 0.761 | True |
| ETTm2 | 720 | relu | 0.414460 | 0.410780 | 0.888 | True |
| ETTm2 | 720 | softsign | 0.415450 | 0.409670 | 1.391 | True |
| ETTm2 | 720 | swish | 0.414056 | 0.410397 | 0.884 | True |
| ETTm2 | 720 | tanh | 0.414484 | 0.408508 | 1.442 | True |
| ETTm2 | 720 | tanh_sin001 | 0.414404 | 0.408527 | 1.418 | True |

## Interpretation

This is a positive-signal probe against `gelu_ffn_linear` as a global default. All three tested cells have a better configuration than default GELU-linear:

- ETTh2/96 prefers output placement over default: `gelu_ffn_outtanh`, +0.640% MSE gain.
- ETTm2/96 prefers activation regime over default: `tanh_sin001_ffn_linear`, +0.194% MSE gain.
- ETTm2/720 prefers regime plus placement: `tanh_ffn_outtanh`, +1.348% MSE gain.

The strongest new signal is placement-dependent: output tanh helps all six activations on ETTh2/96 and ETTm2/720, but hurts all six activations on ETTm2/96. That means placement is not universally good, but it can flip GELU-linear from best-looking default to suboptimal in selected cells.

Claim boundary: one seed only. This supports running targeted confirmation seeds on positive cells, not a final paper-level iTransformer conclusion.
