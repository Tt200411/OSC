# PatchTST Activation Transfer Results

## Scope

This experiment tests whether the Informer activation findings transfer to PatchTST under cloud RTX 4090 execution.

Completed runs:

| Stage | Cells | Configs | Seeds | Seed-runs | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Stage1 FFN activation transfer | 12 | 4 | 3 | 144/144 | complete |
| Stage2 output-tanh placement | 6 | 3 | 3 | 54/54 | complete |

Datasets/horizons:

- ETTh2: 168, 336, 720
- ETTm2: 96, 288, 672
- Solar2, Solar3, Solar5: 24, 96

Stage1 configs:

- `gelu_ffn_linear`
- `tanh_ffn_linear`
- `softsign_ffn_linear`
- `tanh_sin001_ffn_linear`

Stage2 configs:

- `gelu_ffn_outtanh`
- `tanh_ffn_outtanh`
- `tanh_sin001_ffn_outtanh`

All retained result directories include `config.json`, `metrics.npy`, `pred.npy`, and `true.npy`.

## Main Finding

PatchTST shows partial, dataset-specific transfer rather than strong architecture-level transfer.

Stage1 best cells:

| Best config | Cells |
| --- | ---: |
| `gelu_ffn_linear` | 7/12 |
| `softsign_ffn_linear` | 3/12 |
| `tanh_sin001_ffn_linear` | 2/12 |
| `tanh_ffn_linear` | 0/12 |

Non-GELU wins concentrate in:

- ETTh2/720: `softsign_ffn_linear`
- Solar2/24: `tanh_sin001_ffn_linear`
- Solar2/96: `softsign_ffn_linear`
- Solar5/24: `tanh_sin001_ffn_linear`
- Solar5/96: `softsign_ffn_linear`

GELU remains best on:

- ETTh2/168, ETTh2/336
- ETTm2/96, ETTm2/288, ETTm2/672
- Solar3/24, Solar3/96

## Static Ranking

| Config | Mean rank | Best cells | Mean MSE |
| --- | ---: | ---: | ---: |
| `gelu_ffn_linear` | 2.25 | 7 | 0.250041 |
| `tanh_ffn_linear` | 2.42 | 0 | 0.251474 |
| `tanh_sin001_ffn_linear` | 2.50 | 2 | 0.251468 |
| `softsign_ffn_linear` | 2.83 | 3 | 0.251787 |

The result does not support the claim that bounded or tanh-family activations are generally better than GELU in PatchTST. It does support a weaker claim: activation choice remains a meaningful design variable, with non-GELU activations improving selected dataset-horizon cells.

## Output-Tanh Placement

Output tanh does not transfer to PatchTST.

Paired against the corresponding linear-head configuration over the six Stage2 cells:

| Base activation | Paired seed-runs | Output-tanh win rate | Mean gain vs linear |
| --- | ---: | ---: | ---: |
| `tanh` | 18 | 0.0% | -5.28% |
| `tanh_sin001` | 18 | 0.0% | -5.31% |
| `gelu` | 18 | 0.0% | -5.44% |

Across all 54 paired seed-runs, output tanh loses every comparison to the matching linear-head variant.

This is an important negative result for the paper story: output bounding appears architecture-dependent. The Informer long-horizon output-tanh signal should not be presented as a general forecasting rule.

## Paper Claim Update

Recommended wording:

> PatchTST transfer is partial. The activation effect does not disappear, but the strongest Informer conclusion does not directly carry over: GELU remains the best static PatchTST choice on 7/12 tested cells, while bounded/tanh-family activations win selected ETTh2 and Solar cells. Output-tanh placement fails consistently in PatchTST, losing all paired comparisons against linear heads.

Claims to avoid:

- "The Informer activation ranking transfers to PatchTST."
- "Tanh-family activations are generally better than GELU in PatchTST."
- "Output tanh is a robust placement strategy across architectures."

Best submission use:

- Use PatchTST as an architecture-transfer stress test.
- State that activation sensitivity is real but architecture-dependent.
- Keep the strong full-grid claim scoped to Informer.
- Treat PatchTST as evidence against over-generalizing placement rules.

## Files

- Stage1 aggregation: `patchtst_remote_results/analysis_patchtst_stage1_actfix/patchtst_claim_summary.md`
- Stage2 aggregation: `patchtst_remote_results/analysis_patchtst_stage2_actfix/patchtst_claim_summary.md`
- Combined transfer report: `patchtst_remote_results/analysis_patchtst_transfer_actfix/patchtst_transfer_claim_summary.md`
