# iTransformer Activation Transfer Results

## Scope

This experiment tests whether the Informer activation findings transfer to iTransformer under cloud RTX 4090 execution.

Completed runs:

| Stage | Datasets | Horizons | Configs | Seeds | Seed-runs | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Stage1 FFN activation transfer | ETTh2, ETTm2 | 96, 336, 720 | 4 | 1 | 24/24 | complete |

Configs:

- `gelu_ffn_linear`
- `tanh_ffn_linear`
- `softsign_ffn_linear`
- `tanh_sin001_ffn_linear`

The run only changes FFN activation. Output-tanh placement is not tested here. All retained rows have `config.json`, `metrics.npy`, `pred.npy`, and `true.npy`.

## Main Finding

iTransformer does not show strong transfer of the Informer tanh-family finding. GELU is best on 5/6 dataset-horizon cells; the only non-GELU best cell is ETTm2/96 with `tanh_sin001_ffn_linear`.

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

## Paired vs GELU

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

## Paper Claim Update

Recommended wording:

> iTransformer provides a stricter architecture-transfer stress test. In a one-seed ETTh2/ETTm2 pilot, GELU remains best on 5/6 dataset-horizon cells, while the tanh-family only wins narrowly on ETTm2/96. This weakens any claim that the Informer tanh-family advantage is architecture-universal, and instead supports the paper's more conservative framing: activation effects are real but mediated by architecture and regime.

Claims to avoid:

- "The Informer activation ranking transfers to iTransformer."
- "Tanh-family activations are generally better than GELU in iTransformer."
- "iTransformer confirms our placement rule." Placement was not tested.

Best submission use:

- Use iTransformer as negative/weak transfer evidence.
- Keep strong empirical claims scoped to Informer full-grid results.
- Use PatchTST and iTransformer together to argue for architecture-dependent transfer, not universal activation replacement.

## Files

- Raw deduplicated results: `itransformer_remote_results/analysis_itransformer_stage1_actfix3/itransformer_protocol_raw_dedup.csv`
- Aggregation: `itransformer_remote_results/analysis_itransformer_stage1_actfix3/itransformer_protocol_aggregate.csv`
- Claim summary: `itransformer_remote_results/analysis_itransformer_stage1_actfix3/itransformer_claim_summary.md`
- Audit: `.aris/itransformer_experiment_audit_20260521.md`
- Result-to-claim: `.aris/itransformer_result_to_claim_20260521.md`
- Paper table: `paper/figures/TABLE_itransformer_transfer.tex`

## Regime/Placement Probe

Follow-up on 2026-05-22: we expanded the iTransformer probe to all currently
implemented FFN regimes crossed with linear vs output-tanh placement on a small
three-cell matrix.

Scope:

- Cells: ETTh2/96, ETTm2/96, ETTm2/720
- Regimes: GELU, ReLU, Swish, tanh, softsign, tanh_sin001
- Placements: FFN + linear output, FFN + output tanh
- Seed: 2024
- Coverage: 36/36 finite rows; 12 reused linear rows and 24 new cloud RTX 4090 runs

Best configurations:

| Dataset | Horizon | Best config | MSE | GELU-linear MSE | Gain vs GELU-linear |
| --- | ---: | --- | ---: | ---: | ---: |
| ETTh2 | 96 | `gelu_ffn_outtanh` | 0.299667 | 0.301598 | 0.640% |
| ETTm2 | 96 | `tanh_sin001_ffn_linear` | 0.183869 | 0.184226 | 0.194% |
| ETTm2 | 720 | `tanh_ffn_outtanh` | 0.408508 | 0.414092 | 1.348% |

This is the first iTransformer probe where every tested cell beats the default
`gelu_ffn_linear` baseline. The signal is explicitly conditional:
output-tanh helps all six activations on ETTh2/96 and ETTm2/720, but hurts all
six activations on ETTm2/96. This supports the paper's conditional-selection
story and gives a targeted set of cells for multi-seed confirmation.

Files:

- Summary: `itransformer_remote_results/analysis_itransformer_placeprobe_20260522/itransformer_placeprobe_summary.md`
- Raw rows: `itransformer_remote_results/analysis_itransformer_placeprobe_20260522/itransformer_placeprobe_all_rows.csv`
- Best vs GELU: `itransformer_remote_results/analysis_itransformer_placeprobe_20260522/itransformer_placeprobe_best_vs_gelu.csv`
- Output placement pairs: `itransformer_remote_results/analysis_itransformer_placeprobe_20260522/itransformer_placeprobe_output_tanh_pairs.csv`
