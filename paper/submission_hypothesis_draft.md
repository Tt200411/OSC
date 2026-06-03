# Submission Hypothesis Draft

Working title:

**Activations Are Not Defaults: Dataset- and Regime-Dependent Nonlinearity Selection for Long-Horizon Time-Series Forecasting**

Status:

- Current evidence: complete Informer Phase-4 full grid.
- New evidence: PatchTST minimal transfer complete; transfer is partial and architecture-dependent.
- New evidence: iTransformer quick transfer complete; direct transfer is weak/negative, with GELU best in 5/6 tested cells.
- Missing before strong submission: prospective factor-router validation.
- Intended use: paper-story draft and experiment checklist, not final camera-ready prose.

---

## Abstract Draft

Activation functions in time-series forecasting models are usually treated as implementation defaults, while research attention is mostly placed on architecture, attention mechanisms, input window design, and loss functions. We challenge this convention by studying activation-only changes under fixed forecasting architectures and protocols. In a completed Informer full-grid study covering 36 dataset-horizon cells, nine activation configurations, and three seeds per configuration, non-GELU activations are best in 34 of 36 cells, while bounded and tanh-family activations achieve the strongest average ranks. These results show that activation choice is a first-order design variable even when attention, depth, width, optimizer, and training schedule are held fixed.

However, the same full grid also rejects a simple universal-activation story: Solar3 remains GELU-best at both tested horizons, output tanh is highly beneficial in selected long-horizon ETT/ETTm cells but poor on average, and decoder-only tanh wins no cell. Transfer probes further sharpen this boundary. A minimal PatchTST study finds non-GELU activations best in 5 of 12 cells, while GELU remains best in 7 cells and output tanh loses all 54 paired comparisons against matched linear heads. A one-seed iTransformer pilot is even more conservative: GELU remains best in 5 of 6 ETTh2/ETTm2 cells, with only a narrow `tanh_sin001` win on ETTm2/96. These results show that activation sensitivity can persist beyond Informer, but activation ranking and placement are architecture-dependent. To move beyond static activation replacement, we develop factor-bin diagnostics inspired by financial regime analysis and ask whether input-window regimes can predict the correct activation family and placement. Current diagnostics reveal substantial regime dependence, but leave-one-seed-out evidence is too localized to support a deployable router.

**TODO before final abstract**: replace the final router sentence with actual prospective factor-router results.

---

## 1. Introduction

Long-horizon forecasting papers often frame progress as architectural: more efficient attention, better patching, decomposition, channel handling, or improved temporal embeddings. In contrast, the nonlinearity inside feed-forward blocks is commonly inherited from the reference implementation. This is a strong implicit assumption. A forecasting model's activation function controls saturation, sign preservation, gradient scale, and sensitivity to large input excursions. These properties are directly relevant to time-series data, where volatility, jumps, trends, reversals, and heavy tails vary across datasets and horizons.

This paper studies a deliberately narrow question:

> If the forecasting architecture, optimizer, data split, horizon-specific window, and training schedule are fixed, can changing only the activation function and its placement materially improve performance?

Our current answer from Informer is yes. In a full-grid study over ETTh1, ETTh2, ETTm1, ETTm2, and eight Solar sites, we evaluate 36 dataset-horizon cells, nine activation configurations, and three seeds per configuration. Non-GELU activations are best in 34 of 36 cells. The best average ranks belong to tanh-family and bounded activations: `tanh_sin001_all`, `tanh_all`, and `softsign_all`. These gains are not confined to one dataset: all 20 ETT/ETTm cells and 14 of 16 Solar cells prefer a non-GELU activation.

At the same time, our evidence argues against a universal activation. The best configuration varies across dataset and horizon. Output tanh wins several long-horizon ETT/ETTm cells but is weak globally. Encoder-side tanh is strongest in selected Solar96 cells. Decoder-only tanh never wins. Solar3 remains GELU-best at both tested horizons. The scientific story is therefore not that we discovered a single replacement for GELU, but that activation choice is conditional.

The natural next question is whether this conditionality can be predicted from the data itself. We explore this using factor-bin diagnostics inspired by financial regime analysis. Input windows are summarized by volatility, realized volatility, jump intensity, trend consistency, max absolute change, tail ratio, reversal rate, mean reversion strength, and related factors. The goal is to determine whether these factors can predict both the activation regime and its placement, enabling low-cost activation selection without modifying the forecasting architecture.

The present evidence is not yet sufficient to claim a deployed zero-shot selector. Summary-level factor bins reveal regime dependence across the full matrix, but the stricter leave-one-seed-out oracle is positive only in Solar1/96. Therefore, the final paper should treat factor bins as a mechanism and selection-design tool until a prospective router experiment is completed.

### Contributions

1. **Activation-only forecasting study.** We isolate activation family and placement in Informer while holding attention, width, depth, optimizer, data splits, and training protocol fixed.
2. **Full-grid evidence.** We complete a three-seed Informer matrix of 972 seed-runs across 36 dataset-horizon cells and nine activation configurations.
3. **Non-universal but strong activation families.** We show that bounded and tanh-family activations are the strongest static candidates, while the complete grid rejects a universal winner.
4. **Placement diagnosis.** We separate full-hidden replacement, encoder-only tanh, decoder-only tanh, and hidden+output tanh, showing that placement effects are dataset- and horizon-specific.
5. **Regime-conditioned selection problem.** We introduce factor-bin diagnostics and define the missing prospective experiment needed to test whether input-window factors can select activation regime and placement.
6. **Architecture-transfer stress tests.** We test PatchTST and iTransformer. PatchTST shows partial transfer, while iTransformer mostly remains GELU-best; together they support architecture-dependent activation selection rather than a universal replacement rule.

---

## 2. Hypotheses

### H1: Activation functions are first-order design variables in forecasting.

Changing only activation family or placement, while holding architecture and protocol fixed, can materially change forecasting performance.

Current status: **supported in Informer**.

Evidence:

- 972/972 canonical Informer seed-runs complete.
- 324/324 dataset-horizon-config cells have three seeds.
- 34/36 dataset-horizon cells are best with non-GELU activations.

### H2: Bounded and tanh-family activations are strong static candidates.

Bounded/sign-preserving nonlinearities and small tanh-centered oscillatory perturbations can outperform GELU-style defaults in many forecasting regimes.

Current status: **supported as static candidates, not as universal winners**.

Current ranking:

| Config | Mean rank | Best cells | Top-3 cells | Mean gain vs GELU | Positive cell rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tanh_sin001_all` | 2.78 | 8 | 27 | 11.7% | 83.3% |
| `tanh_all` | 3.19 | 7 | 25 | 11.9% | 80.6% |
| `softsign_all` | 3.47 | 8 | 22 | 11.2% | 80.6% |

### H3: Oscillatory perturbations help in some regimes, but are not universally tied to volatility.

Current status: **partially supported**.

Supported:

- `tanh_sin001_all` has the best mean rank across the full Informer grid.

Not yet supported:

- A direct rule such as "high volatility implies tanh_sin" or "high volatility implies oscillatory activation."

Required evidence:

- Prospective factor-router validation on held-out cells or windows.

### H4: Placement matters and can reverse the effect of the same activation.

Current status: **supported in Informer**.

| Placement | Best cells | Current interpretation |
| --- | ---: | --- |
| Full hidden activation | 28/36 | Most stable default placement |
| Hidden + output tanh | 5/36 | Strong in selected long-horizon ETT/ETTm, risky globally |
| Encoder-only tanh | 3/36 | Solar96-specific signal |
| Decoder-only tanh | 0/36 | Not a promising primary placement |

### H5: Data factors can predict activation regime and placement.

Current status: **open; highest-priority missing experiment**.

Current factor-bin diagnostics show regime dependence, but do not yet prove prediction. The final claim requires a prospective rule that uses input-window factors before seeing test errors.

---

## 3. Method Draft

### 3.1 Activation-only intervention

We define an activation-only intervention as changing the nonlinearity used inside the forecasting model while keeping architecture, attention, width, depth, optimizer, training schedule, data split, and horizon-specific input/output windows fixed.

In Informer, we test the following configurations:

| Config | Encoder FFN | Decoder FFN | Output | Purpose |
| --- | --- | --- | --- | --- |
| `gelu_all` | GELU | GELU | Linear | Same-protocol baseline |
| `relu_all` | ReLU | ReLU | Linear | Monotone non-bounded control |
| `swish_all` | Swish | Swish | Linear | Smooth non-bounded control |
| `tanh_all` | tanh | tanh | Linear | Bounded sign-preserving candidate |
| `softsign_all` | softsign | softsign | Linear | Bounded sign-preserving candidate |
| `tanh_sin001_all` | tanh + 0.01 sin | tanh + 0.01 sin | Linear | Small oscillatory perturbation |
| `enc_tanh_dec_gelu` | tanh | GELU | Linear | Encoder-placement test |
| `enc_gelu_dec_tanh` | GELU | tanh | Linear | Decoder-placement test |
| `tanh_all_outtanh` | tanh | tanh | tanh | Hidden + output bounding test |

### 3.2 Factor-bin diagnostics

For each input window, compute regime factors:

- Volatility and realized volatility
- Volatility shock
- Max absolute change
- Jump intensity
- Tail ratio
- Trend and trend consistency
- Reversal rate
- Mean reversion strength
- Range, skewness, kurtosis, downside/upside volatility

We use two diagnostic levels:

1. **Summary-level factor bins.** Dataset-horizon cells are grouped by target-window factor bins. This covers the full matrix but is not a deployable selector.
2. **Sample-level oracle.** Saved prediction arrays are evaluated per sample. Same-split oracle is an upper bound. Leave-one-seed-out oracle is a weak generalization test.

### 3.3 Prospective factor-router design

This is the missing mechanism experiment.

The router must obey:

- It can use only input-window factors.
- It cannot use test errors to choose an activation.
- It must choose both activation regime and placement before evaluation.
- It must be compared against GELU, best static activation, and oracle upper bound.

Candidate router forms:

| Router | Description | Claim strength if successful |
| --- | --- | --- |
| Rule-based threshold | Hand-designed thresholds on volatility/jump/trend factors | Weak but interpretable |
| Cell-trained decision tree | Learn factor-to-activation rules on training cells | Moderate |
| Leave-dataset-out router | Train on all but one dataset/site, test on held-out site | Strong |
| Leave-horizon-out router | Train on some horizons, test on unseen horizon | Strong |
| Meta-router with uncertainty | Choose static activation unless confidence is high | Most practical |

---

## 4. Current Completed Results

### 4.1 Informer full-grid coverage

| Group | Dataset-horizon cells | Config cells | Seed-runs | Complete three-seed cells |
| --- | ---: | ---: | ---: | ---: |
| ETTh | 10 | 90 | 270/270 | 90/90 |
| ETTm | 10 | 90 | 270/270 | 90/90 |
| Solar | 16 | 144 | 432/432 | 144/144 |
| All | 36 | 324 | 972/972 | 324/324 |

### 4.2 Informer main result

| Result | Value |
| --- | ---: |
| Non-GELU best cells | 34/36 |
| GELU-best cells | 2/36 |
| ETTh/ETTm non-GELU best cells | 20/20 |
| Solar non-GELU best cells | 14/16 |
| Best static families | `tanh_sin001_all`, `tanh_all`, `softsign_all` |

GELU-best counterexamples:

| Dataset | Horizon | Best config |
| --- | ---: | --- |
| Solar3 | 24 | GELU |
| Solar3 | 96 | GELU |

### 4.3 Placement result

| Placement | Winning cells | Key cells |
| --- | ---: | --- |
| Full hidden | 28 | Most ETTh/ETTm/Solar cells |
| Hidden + output tanh | 5 | ETTh1/720, ETTh2/336, ETTh2/720, ETTm2/288, ETTm2/672 |
| Encoder-only tanh | 3 | Solar4/96, Solar5/96, Solar7/96 |
| Decoder-only tanh | 0 | None |

### 4.4 Current factor-bin result

| Diagnostic | Coverage | Current conclusion |
| --- | ---: | --- |
| Summary-level factor bins | 36 cells, 783 rows | Regime dependence exists |
| Factor-conditioned contrasts | 196 strong contrasts | Useful for mechanism diagnosis |
| Same-split oracle | Array-ready cells only | Diagnostic upper bound |
| LOSO oracle | 84 rows | Only 7 positive rows, all Solar1/96 |

Interpretation:

Factor bins are promising as a diagnostic and experiment-design tool, but not yet a validated zero-shot selector.

---

## 5. Missing Experiments Before Strong Submission

The user's current prioritization was correct:

1. PatchTST transfer: completed; result is partial, not a broad positive transfer.
2. Factor-bin mechanism: still the key missing experiment; can input-window factors actually predict activation regime and placement?

PatchTST narrows the story: the paper is no longer only an Informer activation audit, but the broad claim must be architecture-sensitive. The remaining decisive gap is whether factor bins can become a prospective selector rather than a diagnostic.

---

## 6. Completed Experiment Matrix A: PatchTST Transfer

### Goal

Test whether activation-family sensitivity transfers from Informer to PatchTST under matched PatchTST protocols.

### Completed minimal matrix

| Model | Dataset | Horizons | Configs | Seeds | Required runs | Status |
| --- | --- | --- | --- | --- | ---: | --- |
| PatchTST | ETTh2 | 168, 336, 720 | `gelu_ffn_linear`, `tanh_ffn_linear`, `softsign_ffn_linear`, `tanh_sin001_ffn_linear` | 2024, 2025, 2026 | 36 | complete |
| PatchTST | ETTm2 | 96, 288, 672 | `gelu_ffn_linear`, `tanh_ffn_linear`, `softsign_ffn_linear`, `tanh_sin001_ffn_linear` | 2024, 2025, 2026 | 36 | complete |
| PatchTST | Solar2 | 24, 96 | `gelu_ffn_linear`, `tanh_ffn_linear`, `softsign_ffn_linear`, `tanh_sin001_ffn_linear` | 2024, 2025, 2026 | 24 | complete |
| PatchTST | Solar3 | 24, 96 | `gelu_ffn_linear`, `tanh_ffn_linear`, `softsign_ffn_linear`, `tanh_sin001_ffn_linear` | 2024, 2025, 2026 | 24 | complete |
| PatchTST | Solar5 | 24, 96 | `gelu_ffn_linear`, `tanh_ffn_linear`, `softsign_ffn_linear`, `tanh_sin001_ffn_linear` | 2024, 2025, 2026 | 24 | complete |

Total minimal transfer runs: **144**.

### Completed placement matrix

| Model | Dataset group | Extra configs | Purpose | Runs | Status |
| --- | --- | --- | --- | ---: | --- |
| PatchTST | ETTh2/ETTm2 long horizons, Solar3/96, Solar5/96 | `gelu_ffn_outtanh`, `tanh_ffn_outtanh`, `tanh_sin001_ffn_outtanh` | Test output-bounding transfer | 54 | complete |

### PatchTST result table

| Model | Dataset | Horizon | Default MSE | Best activation | Best MSE | Gain vs default | Transfer verdict |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| PatchTST | ETTh2 | 168 | 0.325880 | `gelu_ffn_linear` | 0.325880 | 0.00% | default |
| PatchTST | ETTh2 | 336 | 0.328450 | `gelu_ffn_linear` | 0.328450 | 0.00% | default |
| PatchTST | ETTh2 | 720 | 0.380881 | `softsign_ffn_linear` | 0.378315 | 0.67% | partial transfer |
| PatchTST | ETTm2 | 96 | 0.165589 | `gelu_ffn_linear` | 0.165589 | 0.00% | default |
| PatchTST | ETTm2 | 288 | 0.262739 | `gelu_ffn_linear` | 0.262739 | 0.00% | default |
| PatchTST | ETTm2 | 672 | 0.355232 | `gelu_ffn_linear` | 0.355232 | 0.00% | default |
| PatchTST | Solar2 | 24 | 0.065450 | `tanh_sin001_ffn_linear` | 0.065173 | 0.42% | partial transfer |
| PatchTST | Solar2 | 96 | 0.099666 | `softsign_ffn_linear` | 0.098550 | 1.12% | partial transfer |
| PatchTST | Solar3 | 24 | 0.279035 | `gelu_ffn_linear` | 0.279035 | 0.00% | default |
| PatchTST | Solar3 | 96 | 0.446409 | `gelu_ffn_linear` | 0.446409 | 0.00% | default |
| PatchTST | Solar5 | 24 | 0.114792 | `tanh_sin001_ffn_linear` | 0.113260 | 1.33% | partial transfer |
| PatchTST | Solar5 | 96 | 0.176366 | `softsign_ffn_linear` | 0.174583 | 1.01% | partial transfer |

### Output-tanh placement result

| Base activation | Paired seed-runs | Output-tanh win rate | Mean gain vs linear |
| --- | ---: | ---: | ---: |
| `tanh` | 18 | 0.0% | -5.28% |
| `tanh_sin001` | 18 | 0.0% | -5.31% |
| `gelu` | 18 | 0.0% | -5.44% |

Output tanh loses all 54 paired seed-run comparisons against the matching linear-head PatchTST configuration.

### Interpretation rules

| Outcome | Paper claim |
| --- | --- |
| PatchTST also shows non-default activation wins in most selected cells | Claim can generalize beyond Informer |
| PatchTST shows smaller but positive bounded/tanh-family signal | Claim becomes "architecture-sensitive but transferable" |
| PatchTST shows no clear effect | Main claim should remain Informer-specific; PatchTST becomes a limitation |
| PatchTST reverses the winners | Stronger story: activation selection is architecture-dependent, not just dataset-dependent |

Observed outcome: **architecture-sensitive but only partial transfer**. Activation sensitivity remains visible, especially on ETTh2/720 and Solar2/Solar5, but the Informer static ranking and output-tanh placement do not directly transfer.

---

## 7. Missing Experiment Matrix B: Prospective Factor Router

### Goal

Test whether factor bins can predict which activation regime and placement should be used before seeing test errors.

### Candidate activations to route over

Keep the action space small enough to validate:

| Action | Meaning | Why included |
| --- | --- | --- |
| `gelu_all` | default baseline | Needed for fallback and counterexamples |
| `tanh_sin001_all` | best mean-rank static config | Strong global candidate |
| `tanh_all` | bounded tanh full-hidden | Strong and interpretable |
| `softsign_all` | bounded softsign full-hidden | Strong and stable |
| `enc_tanh_dec_gelu` | encoder-only tanh | Solar96 placement candidate |
| `tanh_all_outtanh` | hidden + output tanh | ETT/ETTm long-horizon placement candidate |

### Factor set

| Factor group | Factors |
| --- | --- |
| Volatility | volatility, realized volatility, volatility shock, downside/upside volatility |
| Jump/tail | max absolute change, jump intensity, tail ratio, kurtosis |
| Trend | trend, trend consistency, cumulative change, momentum |
| Reversal/reversion | reversal rate, mean crossing rate, mean reversion strength |
| Shape | range, skewness, max drawdown, max drawup, direction entropy |

### Router validation matrix

| Validation split | Train source | Test target | Required output | Status |
| --- | --- | --- | --- | --- |
| Leave-one-seed-out | Two seeds in same cell | Held-out seed | Router MSE vs best static | TODO |
| Leave-one-site-out Solar | Solar sites except one | Held-out Solar site | Generalization across sites | TODO |
| Leave-one-dataset-family-out | ETT train, Solar test or inverse | Held-out family | Cross-family robustness | TODO |
| Leave-one-horizon-out | Some horizons | Unseen horizon | Horizon transfer | TODO |
| Future-window split | Earlier test windows | Later test windows | Time-forward robustness | TODO |

### Router result table placeholder

| Split | Dataset/horizon scope | GELU MSE | Best static MSE | Same-split oracle MSE | Prospective router MSE | Router gain vs best static | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| LOSO seed | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Leave-one-Solar-site | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Leave-one-horizon | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Future-window | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

### Router mechanism table placeholder

| Factor condition | Selected activation | Selected placement | Training evidence | Held-out evidence | Use in final paper? |
| --- | --- | --- | --- | --- | --- |
| High volatility + high max_abs_change | TODO | TODO | TODO | TODO | TODO |
| High trend consistency | TODO | TODO | TODO | TODO | TODO |
| High jump intensity | TODO | TODO | TODO | TODO | TODO |
| High reversal rate | TODO | TODO | TODO | TODO | TODO |
| Low volatility + stable trend | TODO | TODO | TODO | TODO | TODO |

### Success criteria

Use conservative criteria:

1. Router must beat GELU.
2. Router should beat or match the best static activation on at least one held-out split.
3. Router should not collapse badly on counterexample cells such as Solar3.
4. Router choices should be interpretable, not just post hoc oracle choices.

### If the router fails

The paper can still be strong if framed correctly:

- Main contribution remains complete activation audit and placement map.
- Factor bins become diagnostic evidence, not an algorithm.
- Future work: learning robust routers from larger cross-architecture matrices.

---

## 8. Proposed Main Paper Structure

### Section 1: Introduction

Main message:

- Forecasting models inherit activation defaults.
- Activation functions encode assumptions about saturation, sign, gradients, and response to extremes.
- We show activation choice matters under fixed architecture.
- We then ask whether factors can predict which activation and placement to use.

### Section 2: Related Work

Subsections:

1. Transformer forecasting architectures
2. Activation functions and bounded nonlinearities
3. Periodic/oscillatory representations
4. Regime-conditioned time-series analysis
5. Model selection and hyperparameter transfer

### Section 3: Activation Regimes and Placement

Define:

- Activation regime
- Placement
- Full-hidden vs encoder-only vs decoder-only vs output activation
- Factor-bin diagnostic setup

### Section 4: Informer Full-Grid Evidence

Completed:

- Coverage table
- Best activation table
- Static ranking table
- Placement analysis

### Section 5: Cross-Architecture Transfer

Completed:

- PatchTST transfer matrix.
- Result is mixed: use as architecture-sensitivity evidence.
- Do not use as evidence that the Informer ranking transfers directly.

### Section 6: Factor-Conditioned Activation Selection

Partially completed:

- Summary-level factor bins.
- Sample-level oracle diagnostics.

TODO:

- Prospective router result.
- Placement/regime prediction table.

### Section 7: Discussion and Limitations

Must include:

- No universal activation.
- No direct Informer Table 2 reproduction claim.
- Current factors are diagnostic unless prospective router succeeds.
- PatchTST shows partial transfer and limits over-generalization beyond Informer.

---

## 9. Claim Ladder for Final Submission

### Safe claim now

Activation choice is a dataset- and horizon-sensitive design variable in Informer forecasting.

### Updated claim after PatchTST and iTransformer transfer

Activation choice can remain relevant beyond Informer, but the best activation family and placement are architecture-dependent. PatchTST gives partial transfer; iTransformer gives weak/negative direct transfer.

### Stronger claim after router validation

Input-window factors can prospectively guide activation regime and placement selection.

### Strongest possible claim, only if both missing experiments succeed

Activation regime and placement can be selected from data characteristics to improve forecasting performance across architectures without modifying the model structure.

---

## 10. Direct Answer To Current Concern

Your current understanding is mostly correct:

1. **PatchTST transfer was the right first missing experiment.** It is now complete and shows partial transfer: enough to claim broader relevance of activation sensitivity, not enough to claim the Informer ranking transfers.
2. **iTransformer is now a second stress test, not a rescue result.** It is complete as a one-seed pilot and mostly favors GELU, so it further supports an architecture-dependent story rather than a universal tanh-family transfer claim.
2. **Factor-bin mechanism is the right second missing experiment.** Current factor bins show regime dependence, but not yet predictive selection. The missing experiment must be prospective: train or define the rule first, then evaluate on held-out seeds/sites/horizons/windows.

The only adjustment is wording:

- Do not say the paper already has a zero-shot activation-selection algorithm.
- Say the current paper has a complete activation map and a factor-conditioned selection hypothesis.
- The final submission becomes substantially stronger if the prospective router matrix is filled.
