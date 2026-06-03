# Paper Plan

**Title**: Where and When Do Bounded Activations Help Informer Forecasting? Placement and Regime Diagnostics under an Official-Script Protocol  
**Venue**: ICLR draft target  
**Type**: empirical/diagnostic method paper  
**Date**: 2026-05-15  
**Page budget**: 9 pages main body to Conclusion, excluding references and appendix  
**Section count**: 6 main sections plus appendix  
**Assurance**: submission requested; compilation and audit gate are currently blocked by missing local LaTeX tools.

## Claims-Evidence Matrix

| Claim | Evidence | Status | Section |
|---|---|---|---|
| Activation choice alone can change Informer forecasting error without changing architecture or parameter scale. | Controlled CLI/config activation switch in `lee_ocil`; 378 synchronized run rows; common Informer shape with activation-only intervention. | Supported | §3, §4 |
| Bounded/tanh-family hidden activations are the strongest current mechanism over GELU. | Phase-2/3 exact-protocol cells: ETTh1/168 `tanh` +13.2%; ETTh1/336 `softsign` +10.0%; ETTh2/168 `tanh_sin` +42.1%; ETTh2/336 `tanh_sin` +44.7%; Solar96 encoder-`tanh` wins placement table. | Supported | §4.1 |
| Explicit `tanh + a sin(x)` perturbation is not globally reliable against `tanh`. | ETTh1/24 a=0.01: -1.66%; ETTh2/168 a=0.01: -2.00%; Solar1/96 a=0.01: -0.42%; Solar5/24 a=0.10: +3.21% isolated positive. | Supported as negative/mixed | §4.2 |
| Factor bins explain conditional structure but do not yet justify a deployed router. | Solar1/96 LOSO factor oracle is positive for max absolute change, volatility shock, and volatility; Solar5/96 has no positive LOSO oracle. | Partially supported | §5.1 |
| Lee-OC has dataset- and horizon-specific behavior beyond pure `tanh`, but is not a safe default. | Lee3 vs `tanh` on ETTh2/168: +22.07%, win 0.833; Lee3 on Solar1/24: -17.20%; Lee1 vs `tanh` mostly ineffective except ETTh2/168. | Supported as conditional | §4.3, §5.2 |
| ReLU and Swish controls do not explain away the bounded/tanh-family advantage. | Phase-3 exact-protocol controls: Swish improves over GELU in several cells but remains worse than the bounded/tanh-family best in ETTh1/168, ETTh1/336, ETTh2/168, ETTh2/336, Solar1/96, and Solar5/96; ReLU is not competitive. | Supported | §4 |
| Placement is part of the mechanism, not an afterthought. | Solar1/96 and Solar5/96 prefer encoder-side `tanh`; Solar96 output `tanh` is harmful, while ETTh2 output `tanh` can help. | Supported | §4, §5 |
| Finance-style regime factors are diagnostic but not a deployed router. | Solar1/96 LOSO oracle positive only for max absolute change, volatility shock, and volatility; Solar5/96 has no positive LOSO factor oracle. | Partially supported | §5 |

## Structure

### §0 Abstract

- **Problem**: Activation functions inside time-series transformers are usually treated as fixed implementation choices, making it unclear whether observed gains from oscillatory activations come from oscillation itself or from simpler bounded activation effects.
- **Approach**: Run a controlled Informer activation-family study over ETT and solar forecasting datasets while keeping architecture fixed, then link sample-level errors to input-window volatility and turbulence factors.
- **Key result**: Bounded/tanh-family hidden activations produce the strongest current gains over GELU, including +42.1% and +44.7% MSE reduction on ETTh2 long-horizon cells, while Solar96 shows that placement matters and favors encoder-side `tanh`.
- **Implication**: Oscillation should be used selectively through factor-aware activation choice rather than as a universal replacement.
- **Estimated length**: 170-220 words.

### §1 Introduction

- **Opening hook**: Long-sequence forecasting models often expose activation functions as minor defaults, but this default can alter error as much as more expensive architectural changes.
- **Gap**: Prior activation experiments do not cleanly separate boundedness, sign preservation, explicit oscillation, and structured Lee-OC behavior under fixed model capacity.
- **Research questions**:
  1. Does activation-only substitution improve Informer forecasts?
  2. Are the gains caused by oscillation specifically, by bounded nonlinearities, or by smooth non-bounded alternatives such as Swish?
  3. Which placement and input-window regimes favor each activation family?
- **Contributions**:
  1. A controlled activation-family interface for Informer.
  2. A multi-dataset empirical comparison of GELU, ReLU, Swish, bounded activations, small sinusoidal perturbations, and Lee-OC probes.
  3. A placement-aware analysis separating encoder, decoder, and output activations.
  4. A factor-conditioned error analysis using finance-style input-window regime factors.
- **Hero figure**: A two-panel summary: activation-only intervention schematic plus bar chart showing bounded activations vs GELU and perturbations vs `tanh`.
- **Estimated length**: 1.25-1.5 pages.

### §2 Related Work

- **Long-sequence time-series transformers**: Informer and transformer-based forecasting.
- **Activation functions and bounded nonlinearities**: GELU, ReLU, tanh, softsign.
- **Oscillatory or periodic activations**: SIREN/Fourier-feature-style motivation, but position this work as empirical activation perturbation rather than implicit neural representation.
- **Regime-conditioned forecasting analysis**: volatility, anomaly, and turbulence-style factors for explaining when methods help.
- **Estimated length**: 1 page.
- **Citation plan**: Use verified references only. Initial required citations: Transformer, Informer, GELU, ReLU, SIREN/Fourier features, ETT dataset if separate from Informer.

### §3 Method and Experimental Protocol

- **Problem formulation**: Multivariate sequence-to-sequence forecasting with input window length 96 and prediction horizon `pred_len`.
- **Activation family**:
  - Baselines: `gelu`, `relu`, `tanh`.
  - Bounded controls: `softsign`, `scaled_tanh`.
  - Perturbations: `base + a sin(w x + phi)`, `base + a cos(w x + phi)`, deterministic random-Fourier perturbation.
  - Lee-OC variants: Lee type 1 and type 3 for the current paper draft.
- **Controlled model setting**: Informer architecture and parameter scale unchanged.
- **Factor system**: define volatility/turbulence and high-minus-low bin contrast.
- **Metrics**: MSE primary, MAE secondary, seed win rate, relative MSE change.
- **Estimated length**: 1.5 pages.

### §4 Main Results

- **§4.1 Bounded sign-preserving activations vs GELU/ReLU/Swish**:
  - Table: dataset/window/activation/MSE/relative change/win rate.
  - Figure: grouped bars for `tanh`, `softsign`, `tanh_sin`, ReLU, and Swish.
- **§4.2 Oscillatory perturbations vs `tanh`**:
  - Show `tanh_sin`, `tanh_cos`, `tanh_rand`.
  - Emphasize mixed global evidence.
- **§4.3 Lee specificity**:
  - Compare Lee1/Lee3 vs `tanh`.
  - Highlight ETTh2/168 Lee3 as strong but dataset-dependent.
- **Estimated length**: 2.5 pages.

### §5 Factor-Conditioned Analysis and Discussion

- **§5.1 Why factor and placement diagnostics matter**:
  - Explain why "all bins beat GELU" is not diagnostic.
  - Use volatility, volatility shock, max absolute change, jump intensity, tail ratio, trend consistency, and mean-reversion factors.
- **§5.2 When activation placement changes**:
  - Solar96 static ranking favors encoder-side `tanh`.
  - Same-split oracles show diagnostic gains, but LOSO evidence is weak and only positive on Solar1/96 for a few factors.
- **§5.3 Limits of the current evidence**:
  - Some high-gradient rows do not produce global wins.
  - Some windows are low-seed or duplicated across remote batches.
  - Dynamic activations need more runs.
- **Estimated length**: 1.25 pages.

### §6 Conclusion

- Activation-only changes can matter substantially.
- The safest current recommendation is bounded/tanh-family hidden activations as robust controls under the matched protocol.
- Oscillatory perturbations, output activations, and Lee-OC should be selected conditionally with placement-aware and factor-aware diagnostics.
- **Estimated length**: 0.5 pages.

### Appendix

- Full run coverage table.
- Additional activation rows with fewer than 3 seeds.
- Factor definitions and binning details.
- Remote training and synchronization details.

## Figure Plan

| ID | Type | Description | Data Source | Priority |
|---|---|---|---|---|
| Fig 1 | Multi-panel bar chart | Main activation effects: bounded activations vs GELU and perturbations vs `tanh`. | `.aris/claim_support_activation_summary_min3seeds.csv` | HIGH |
| Fig 2 | Heatmap/table plot | Activation-specific relative MSE changes across dataset/window cells. | `.aris/tanh_centered_activation_summary_current.csv`, `.aris/gelu_centered_activation_summary_current.csv` | HIGH |
| Fig 3 | Factor-gradient bar chart | High-minus-low volatility/turbulence target improvement for selected perturbations. | `.aris/factor_bin_high_low_lee_tanh.csv` | HIGH |
| Table 1 | LaTeX table | Dataset/window coverage and seed counts. | `.aris/dataset_coverage_current.csv` | HIGH |
| Table 2 | LaTeX table | Main bounded/tanh-family results vs GELU/ReLU/Swish. | `phase3_remote_results/analysis_current/phase3_combined_protocol_aggregate.csv` | HIGH |
| Table 3 | LaTeX table | Placement comparison: hidden, encoder-only, decoder-only, output. | `phase3_remote_results/analysis_current/phase3_combined_protocol_aggregate.csv` | HIGH |
| Table 4 | LaTeX table | Solar96 static ranking and LOSO factor-oracle diagnostics. | `.aris/phase3_solar96_factor_oracle_20260517/oracle/` | HIGH |

Manual figure need:

- A clean method schematic explaining activation-only intervention and factor-conditioned analysis. The data plots can be generated automatically, but the method schematic should be manually polished or generated later through ARIS illustration tooling.

## Citation Plan

- §1: Transformer, Informer, GELU, activation-function motivation.
- §2: Transformer forecasting, Informer/ETT, activation functions, periodic activations, forecasting-regime analysis.
- §3: Informer and activation references.
- §4-§5: Mostly self-contained empirical results; cite datasets/model setup only.

No citation should be inserted unless the BibTeX entry is verified.

## DATA_NEEDED

- Final de-duplicated aggregation policy: decide whether repeated same seed/config rows are averaged, filtered by run ID, or treated as independent remote repeats.
- Complete `pred_len=336` to at least 3 seeds if long-horizon claims remain central. **Phase-2/3 now has 3-seed ETT 336 cells under the current filtered policy.**
- More dynamic activation runs if the adaptive method becomes a main contribution.
- Statistical confidence intervals after de-duplication.
- Manual method schematic for Fig. 1 if not using ARIS figure-spec.

## Next Steps

- [x] Create `NARRATIVE_REPORT.md`.
- [x] Create `PAPER_PLAN.md`.
- [x] Run paper-figure/table update for Phase-2/3 main tables.
- [x] Run paper-write update for `paper/`.
- [ ] Compile updated paper.
- [ ] Run submission assurance audits after compile and citation finalization.
