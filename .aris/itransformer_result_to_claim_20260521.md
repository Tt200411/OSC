# iTransformer Result-To-Claim Summary

**Date**: 2026-05-21
**Status**: local judgment, pending external review.

## Intended Claim Tested

Whether the Informer activation findings transfer to iTransformer when only the FFN activation is changed.

## Evidence

- Matrix: ETTh2/ETTm2 x horizons 96/336/720 x 4 FFN activations x seed2024 = 24 rows.
- Completed: 24/24 finite rows; 22 new cloud RTX 4090 runs plus 2 reused smoke rows.
- Best cells with non-GELU activation: 1/6.

## Best Cells

| dataset | pred_len | config_name | mse_mean | mae_mean |
| --- | --- | --- | --- | --- |
| ETTh2 | 96 | gelu_ffn_linear | 0.301598 | 0.351459 |
| ETTh2 | 336 | gelu_ffn_linear | 0.422894 | 0.432135 |
| ETTh2 | 720 | gelu_ffn_linear | 0.428986 | 0.447237 |
| ETTm2 | 96 | tanh_sin001_ffn_linear | 0.183869 | 0.268229 |
| ETTm2 | 336 | gelu_ffn_linear | 0.313418 | 0.351346 |
| ETTm2 | 720 | gelu_ffn_linear | 0.414092 | 0.406538 |

## Static Ranking

| config_name | mean_rank | best_cells | mean_mse |
| --- | --- | --- | --- |
| gelu_ffn_linear | 1.50 | 5 | 0.344202 |
| softsign_ffn_linear | 2.83 | 0 | 0.346652 |
| tanh_ffn_linear | 2.83 | 0 | 0.347098 |
| tanh_sin001_ffn_linear | 2.83 | 1 | 0.347102 |

## Paired Deltas vs GELU

| dataset | pred_len | config_name | mse | gelu_mse | gain_vs_gelu_pct | win_vs_gelu |
| --- | --- | --- | --- | --- | --- | --- |
| ETTh2 | 96 | softsign_ffn_linear | 0.302702 | 0.301598 | -0.366 | False |
| ETTh2 | 96 | tanh_ffn_linear | 0.302767 | 0.301598 | -0.388 | False |
| ETTh2 | 96 | tanh_sin001_ffn_linear | 0.302773 | 0.301598 | -0.390 | False |
| ETTh2 | 336 | softsign_ffn_linear | 0.428267 | 0.422894 | -1.271 | False |
| ETTh2 | 336 | tanh_ffn_linear | 0.429384 | 0.422894 | -1.535 | False |
| ETTh2 | 336 | tanh_sin001_ffn_linear | 0.429419 | 0.422894 | -1.543 | False |
| ETTh2 | 720 | softsign_ffn_linear | 0.433635 | 0.428986 | -1.084 | False |
| ETTh2 | 720 | tanh_ffn_linear | 0.436503 | 0.428986 | -1.752 | False |
| ETTh2 | 720 | tanh_sin001_ffn_linear | 0.436578 | 0.428986 | -1.770 | False |
| ETTm2 | 96 | softsign_ffn_linear | 0.184213 | 0.184226 | 0.007 | True |
| ETTm2 | 96 | tanh_ffn_linear | 0.183878 | 0.184226 | 0.189 | True |
| ETTm2 | 96 | tanh_sin001_ffn_linear | 0.183869 | 0.184226 | 0.194 | True |
| ETTm2 | 336 | softsign_ffn_linear | 0.315647 | 0.313418 | -0.711 | False |
| ETTm2 | 336 | tanh_ffn_linear | 0.315574 | 0.313418 | -0.688 | False |
| ETTm2 | 336 | tanh_sin001_ffn_linear | 0.315571 | 0.313418 | -0.687 | False |
| ETTm2 | 720 | softsign_ffn_linear | 0.415450 | 0.414092 | -0.328 | False |
| ETTm2 | 720 | tanh_ffn_linear | 0.414484 | 0.414092 | -0.095 | False |
| ETTm2 | 720 | tanh_sin001_ffn_linear | 0.414404 | 0.414092 | -0.076 | False |

## Verdict

- `claim_supported`: no for direct transfer; partial for activation sensitivity.
- `what_results_support`: iTransformer is sensitive to activation choice, but GELU is the best static choice in 5/6 tested dataset-horizon cells. The only non-GELU win is ETTm2/96, where `tanh_sin001_ffn_linear` improves MSE over GELU by about 0.194%.
- `what_results_dont_support`: the Informer tanh-family advantage does not directly migrate to iTransformer under this protocol. No placement/output-tanh conclusion is tested.
- `missing_evidence`: seeds 2025/2026; output placement variants if we want to test placement; factor/regime features computed specifically against iTransformer residuals.
- `suggested_claim_revision`: present iTransformer as a second negative/weak transfer stress test. The safest paper story is architecture-dependent activation behavior: strong in Informer, mixed/weak in PatchTST, and mostly GELU-favoring in iTransformer.
- `confidence`: medium for the pilot trend, low for any general cross-architecture claim because this is one seed.
