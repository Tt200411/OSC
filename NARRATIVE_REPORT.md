# Narrative Report: Oscillatory and Bounded Activations for Informer Forecasting

**Project date**: 2026-05-15  
**Workflow**: ARIS Workflow 3 input narrative for `/paper-writing "NARRATIVE_REPORT.md" -- effort: max -- assurance: submission`  
**Current paper scope**: ETTh1, ETTh2, Solar1, Solar5. The legacy `Solar` single-station smoke rows are retained as provenance but should not anchor paper claims.

## One-Sentence Story

Keeping Informer architecture and parameter scale fixed, activation effectiveness is best explained jointly by activation family, placement, and input-window regime: bounded/tanh-family hidden activations beat GELU, ReLU, and Swish in current exact-protocol cells, while Solar96 favors encoder-side boundedness and only weakly supports factor-based routing.

## What Changed in the Model

The Informer backbone is kept fixed: attention type, embedding, encoder/decoder depth, hidden width, and training entry points remain unchanged. The controlled variable is the activation used inside the encoder and decoder feed-forward/convolutional blocks.

Implemented activation families:

- `gelu`: pure GELU baseline.
- `relu`: pure ReLU baseline.
- `swish` / `silu`: smooth non-bounded baseline.
- `tanh`, `softsign`, `scaled_tanh`: bounded sign-preserving controls.
- `gelu_sin`, `relu_sin`, `tanh_sin`: base activation plus `a * sin(w*x + phi)`.
- `tanh_cos`: `tanh(x) + a * cos(w*x + phi)`.
- `tanh_rand`: deterministic fixed random-Fourier perturbation around `tanh`.
- `lee`: Lee-OC activation with `lee_type` variants, currently emphasized at type 1 and type 3.
- `dynamic_gelu_sin`: input-window factor-conditioned amplitude rule, implemented but not yet central to the current paper claims.

Relevant code paths:

- Activation implementation: `lee_ocil/models/activations.py`
- Encoder/decoder activation injection: `lee_ocil/models/encoder.py`, `lee_ocil/models/decoder.py`, `lee_ocil/models/model.py`
- Training/config entry point: `lee_ocil/main_informer.py`, `lee_ocil/exp/exp_config.py`, `lee_ocil/exp/exp_informer.py`
- Remote experiment runners: `lee_ocil/scripts/run_phase1_*.sh`, `lee_ocil/scripts/run_phase2_original_protocol_probe.sh`
- Result summarization: `lee_ocil/scripts/summarize_phase1_results.py`
- Factor linkage: `lee_ocil/scripts/analyze_factor_effects.py`
- Hypothesis report helper: `lee_ocil/scripts/report_phase1_hypotheses.py`

## Dataset Screening and Current Evidence Base

The current writing set uses datasets with enough completed rows to support a first paper draft:

| Dataset | Forecast windows currently used | Main target | Current role |
|---|---:|---|---|
| ETTh1 | 24, 168, 336 | OT | Lower-load ETT benchmark; useful for separating short vs long horizon behavior. |
| ETTh2 | 24, 168, 336 | OT | Higher-load and harder ETT benchmark; current strongest Lee-specific signal occurs here. |
| Solar1 | 24, 96 | Power | Solar station 1, requested focus dataset; shows strong bounded-activation gains over GELU. |
| Solar5 | 24, 96 | Power | Solar station 5, requested focus dataset; shows strong bounded-activation gains and some isolated oscillatory wins. |

Current synchronized summary coverage from `phase1_remote_results/all_servers_current/phase1_detailed_results.csv`:

- 378 run records across 6 synchronized servers.
- Main completed multi-seed cells:
  - ETTh1 `pred_len=24`: 3 seeds, 7 activation types.
  - ETTh1 `pred_len=168`: 6 seeds across 4 servers, 5 activation types.
  - ETTh2 `pred_len=168`: 6 seeds across 3 servers, 5 activation types.
  - Solar1 `pred_len=24,96`: 6 seeds across 2 servers, 7 activation types.
  - Solar5 `pred_len=24,96`: 6 seeds across 2 servers, 6 activation types.
- Low-seed cells exist for ETTh1/ETTh2 `pred_len=336` and ETTh2/Solar smoke settings. These are useful as signals only, not final claims.

Raw result paths:

- All-server summary: `phase1_remote_results/all_servers_current/phase1_activation_summary.csv`
- All-server detailed rows: `phase1_remote_results/all_servers_current/phase1_detailed_results.csv`
- Lee-vs-tanh summary: `phase1_remote_results/all_servers_current_tanh_baseline/phase1_activation_summary.csv`
- Per-server prediction arrays/config/logs: `phase1_remote_results/<server_ip>/results/`, `phase1_remote_results/<server_ip>/logs/`
- Current factor contrast tables: `.aris/all_servers_factor_contrast_default.csv`, `.aris/all_servers_factor_contrast_lee_tanh.csv`
- High-low factor-bin evidence tables: `.aris/factor_bin_high_low_default.csv`, `.aris/factor_bin_high_low_lee_tanh.csv`

## Experimental Setup

Common Informer configuration in completed phase-1 runs:

- Features: `M` for multivariate forecasting.
- Sequence length: `seq_len=96`.
- Label length: `label_len=48`.
- Model dimension: `d_model=512`.
- Heads: `n_heads=8`.
- Encoder layers: `e_layers=2`.
- Decoder layers: `d_layers=1`.
- Feed-forward dimension: `d_ff=2048`.
- Training epochs: commonly 6 for the current phase-1 evidence.
- Metrics: MSE primary, MAE secondary.
- Seeds: `2024`, `2025`, `2026` are the core repeated seeds; some expanded long-window rows include additional seed/server repetitions.
- Servers: remote 4090 machines, with `server_ip` preserved in summaries.

Factor analysis:

- Factors include `volatility`, `anomaly_density_intensity`, `turbulence_score`, `mean`, `std`, `cv`, `trend`, `range`, `mean_abs_change`, `max_abs_change`, `skewness`, and `kurtosis`.
- ETTh factor windows use 24 points/day.
- Solar factor windows use 96 points/day.
- Dynamic or explanatory factors use input-window factors only for train-time-compatible interpretation; target-window factors are for post-hoc analysis.
- The analysis emphasizes high-minus-low bin gradients, slope/Spearman, and aggregate target MSE, not merely whether every bin beats GELU.

## Earlier Phase-1 Results

The rows below are retained as provenance for the first draft. The current paper tables should use the Phase-2/3 exact-protocol addendum at the end of this file.

### Claim 1: Bounded sign-preserving activations explain the strongest current gains over GELU.

Evidence from rows with at least 3 seeds:

| Dataset/window | Activation | Baseline | Mean MSE | Relative MSE change | Win rate | Label |
|---|---:|---|---:|---:|---:|---|
| ETTh1 / 24 | `tanh` | GELU | 0.6869 | +6.26% | 0.778 | effective |
| ETTh1 / 24 | `softsign` | GELU | 0.7122 | +4.61% | 0.667 | effective |
| Solar1 / 24 | `tanh` | GELU | 0.1648 | +5.14% | 1.000 | strong effective |
| Solar1 / 96 | `tanh` | GELU | 0.2509 | +12.84% | 1.000 | strong effective |
| Solar1 / 96 | `softsign` | GELU | 0.2429 | +15.71% | 1.000 | strong effective |
| Solar5 / 24 | `tanh` | GELU | 0.1156 | +10.48% | 1.000 | strong effective |
| Solar5 / 96 | `tanh` | GELU | 0.2038 | +14.39% | 1.000 | strong effective |
| Solar5 / 96 | `softsign` | GELU | 0.2046 | +13.34% | 1.000 | strong effective |

Interpretation: The paper can claim that a bounded sign-preserving substitution is currently the best-supported mechanism. It should not claim that Lee-OC alone is the only cause of improvement.

### Claim 2: `tanh + a sin(x)` is not globally reliable against `tanh`, but has isolated useful cells.

Evidence from rows with at least 3 seeds:

| Dataset/window | Perturbation | Baseline | Relative MSE change | Win rate | Label |
|---|---:|---|---:|---:|---|
| ETTh1 / 24 | `tanh_sin`, a=0.01 | `tanh` | -1.66% | 0.167 | ineffective |
| ETTh1 / 168 | `tanh_sin`, a=0.01 | `tanh` | +0.60% | 0.533 | ineffective |
| ETTh2 / 168 | `tanh_sin`, a=0.01 | `tanh` | -2.00% | 0.417 | negative |
| Solar1 / 24 | `tanh_sin`, a=0.01 | `tanh` | +0.48% | 0.667 | ineffective |
| Solar1 / 96 | `tanh_sin`, a=0.01 | `tanh` | -0.42% | 0.333 | ineffective |
| Solar5 / 24 | `tanh_sin`, a=0.10 | `tanh` | +3.21% | 1.000 | effective |
| Solar5 / 96 | `tanh_sin`, a=0.01 | `tanh` | -0.94% | 0.667 | ineffective |
| Solar5 / 96 | `tanh_sin`, a=0.10 | `tanh` | -2.68% | 0.167 | negative |

Interpretation: The current paper should frame explicit oscillation as regime- and parameter-conditioned. The strongest honest claim is not "sin helps everywhere"; it is "small oscillations sometimes help in selected long-horizon or solar regimes, but require factor-aware selection."

### Claim 3: Factor-conditioned analysis is meaningful, but only as a conditional mechanism.

Current positive factor-gradient examples:

- ETTh1 / 168 / `tanh_sin` a=0.01 vs `tanh`:
  - `turbulence_score` high-minus-low target relative MSE changes appear positive on multiple server slices, e.g. +2.84 and +3.74 percentage points, with high-minus-low target win-rate increases of +10.5 and +18.4 points.
  - `volatility` is less stable, including a negative server slice.
- Solar5 / 96 / `tanh_sin` a=0.01 vs `tanh`:
  - `turbulence_score` high-minus-low target relative MSE change appears positive in current factor tables, around +8.68 to +15.03 percentage points depending on server slice, while global MSE is still not improved.
  - This is useful evidence for conditional behavior, not a global improvement claim.
- ETTh2 / 168 / Lee3 vs `tanh`:
  - Global relative MSE change is +22.07% with 0.833 win rate.
  - High-minus-low volatility target relative MSE change is positive in current factor tables, around +9.60 to +11.39 percentage points.

Interpretation: The factor story should be written as: "factor bins reveal where a perturbation is more likely to help, even when global averages are mixed." The paper should avoid claiming that high volatility universally improves every oscillatory perturbation.

### Claim 4: Lee has some special behavior, but not as a stable default.

Lee vs `tanh` evidence:

| Dataset/window | Lee type | Mean MSE | Relative MSE change vs tanh | Win rate | Label |
|---|---:|---:|---:|---:|---|
| ETTh1 / 24 | 1 | 0.6832 | +0.65% | 0.400 | ineffective |
| ETTh1 / 24 | 3 | 0.7648 | -13.95% | 0.000 | negative |
| ETTh1 / 168 | 1 | 1.0608 | +0.78% | 0.833 | ineffective |
| ETTh1 / 168 | 3 | 1.1258 | -5.41% | 0.333 | negative |
| ETTh2 / 168 | 1 | 3.4669 | +4.97% | 0.833 | effective |
| ETTh2 / 168 | 3 | 2.7308 | +22.07% | 0.833 | strong effective |
| Solar1 / 24 | 1 | 0.1625 | +0.58% | 0.667 | ineffective |
| Solar1 / 24 | 3 | 0.1918 | -17.20% | 0.000 | negative |
| Solar1 / 96 | 1 | 0.2538 | -1.32% | 0.000 | ineffective |

Interpretation: Lee3 has a strong ETTh2 long-window signal, but it is negative on Solar1 / 24 and ETTh1 / 168. Lee should be positioned as a candidate structured perturbation with data-dependent behavior, not as a universally superior activation.

## Claims We Can Support

Supported now:

1. Changing only the activation in Informer feed-forward blocks can materially affect forecasting error while preserving architecture and parameter scale.
2. Bounded sign-preserving activations (`tanh`, `softsign`, and sometimes Lee1) explain much of the observed improvement over GELU.
3. Explicit oscillatory perturbations are not globally reliable against their own bounded baseline.
4. Factor-linked analysis is necessary: some perturbations become more favorable in high-volatility or high-turbulence bins even when global averages are mixed.
5. Lee-OC has dataset- and horizon-specific behavior; Lee3 is promising on ETTh2 / 168 but not generally safe.

Not yet supported strongly enough:

1. "`a=0.01` sinusoidal perturbation improves all long high-volatility sequences."
2. "Lee1 is consistently better than `tanh`."
3. "Lee3 is generally better than Lee1."
4. "Random or cosine perturbations are broadly useful." Current evidence is mixed and sparse.
5. "Dynamic factor-conditioned activation improves final test MSE." Implementation exists, but current paper evidence is not yet sufficient.

## Existing Figures, Tables, and Result Artifacts

Existing project figures:

- `Daily_Factors_Analysis/all_datasets_comparison.png`
- `Daily_Factors_Analysis/summary_bar_comparison.png`
- `Daily_Factors_Analysis/summary_correlation_comparison.png`
- `Daily_Factors_Analysis/summary_h_vs_m_comparison.png`
- `Daily_Factors_Analysis/summary_seasonal_comparison.png`
- `Data Visualization/ETTh1_comparison_all_segments.png`
- `Data Visualization/ETTh2_comparison_all_segments.png`
- `normalized_prediction_comparison_100.png`
- `normalized_prediction_comparison_100_minus02.png`
- `prediction_comparison_1000.png`

New paper figures/tables should be generated from:

- `.aris/claim_support_activation_summary_min3seeds.csv`
- `.aris/gelu_centered_activation_summary_current.csv`
- `.aris/tanh_centered_activation_summary_current.csv`
- `.aris/factor_bin_high_low_default.csv`
- `.aris/factor_bin_high_low_lee_tanh.csv`
- `phase1_remote_results/all_servers_current*/phase1_activation_summary.csv`

## Limitations and DATA_NEEDED

DATA_NEEDED before a strong submission claim:

1. Complete and de-duplicate the exact final experiment table. Current summaries include repeated run records from multiple remote batches; paper tables must state whether they aggregate by run row or unique seed/config.
2. Add or finish dynamic factor-conditioned activation experiments if the paper wants to claim adaptive activation selection.
3. Increase `pred_len=336` evidence to at least 3 seeds for ETTh1/ETTh2 if long-horizon claims are central.
4. Run statistical tests or confidence intervals after final de-duplication.
5. Decide whether the paper's main baseline is GELU or `tanh`. The current scientific story needs both: GELU for "activation substitution matters" and `tanh` for "oscillation/Lee specificity."
6. Create a manual architecture/method diagram if needed; ARIS `paper-figure` only covers data-driven plots.

## Recommended Paper Framing

Working title:

**When Do Oscillatory Activations Help Time-Series Transformers? A Factor-Conditioned Study of Informer Forecasting**

Preferred framing:

- This is an empirical/diagnostic method paper.
- The novelty is not just a new activation; it is a controlled activation-family study plus factor-conditioned error analysis.
- The strongest reviewer-safe contribution is the separation between:
  1. bounded sign-preserving effects,
  2. explicit oscillatory perturbation effects,
  3. Lee-OC structured perturbation effects,
  4. input-window data-state conditions.

Avoid overclaiming:

- Do not say "oscillation always improves prediction."
- Do not say "Lee is universally better."
- Do not say "high volatility guarantees improvement."

## Phase-3 Update: ReLU/Swish, Placement, and Solar96 Oracle

Phase-3 added exact-protocol ReLU and Swish controls on cloud RTX 4090s and reran Solar96 factor/oracle diagnostics with a more finance-oriented factor set.

Main protocol-control result:

| Dataset/window | GELU | ReLU | Swish | Best bounded/tanh-family | Relative gain vs GELU |
|---|---:|---:|---:|---:|---:|
| ETTh1 / 168 | 1.046 | 1.063 | 1.022 | `tanh` 0.908 | +13.2% |
| ETTh1 / 336 | 1.355 | 1.378 | 1.296 | `softsign` 1.219 | +10.0% |
| ETTh2 / 168 | 5.142 | 6.726 | 4.004 | `tanh_sin` 2.976 | +42.1% |
| ETTh2 / 336 | 3.928 | 4.219 | 3.074 | `tanh_sin` 2.172 | +44.7% |
| Solar1 / 96 | 0.230 | 0.236 | 0.226 | encoder-`tanh` 0.224 | +2.5% |
| Solar5 / 96 | 0.195 | 0.198 | 0.194 | encoder-`tanh` 0.180 | +7.6% |

Interpretation: Swish is a strong non-bounded smooth baseline, but it does not explain away the bounded/tanh-family advantage. ReLU is not competitive. Placement remains central: Solar96 prefers encoder-side boundedness, while ETTh2 sometimes benefits from output `tanh`.

Solar96 oracle result:

- Solar1/96 static target-channel ranking: encoder-`tanh` 0.177558, full `tanh` 0.177967, `tanh_sin` 0.183026, GELU 0.192832.
- Solar5/96 static target-channel ranking: encoder-`tanh` 0.250646, `tanh_sin` 0.255040, full `tanh` 0.263984, GELU 0.281201.
- LOSO factor-oracle gains are positive only on Solar1/96 for `max_abs_change` (+1.57%), `volatility_shock` (+1.26%), and `volatility` (+0.92%). Solar5/96 has no positive LOSO factor oracle.

Updated paper story: activation choice is not a universal `tanh` claim. The next version should argue that activation effectiveness is jointly determined by data regime and placement. Finance-style factors are useful diagnostics for where to test activation placement, but current evidence is not strong enough for a deployable router claim.
