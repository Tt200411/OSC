# Pairwise Factor Router Method Report

Date: 2026-05-23

## Motivation from Prior Work

The router problem is closer to algorithm selection and time-series
meta-learning than to ordinary multi-class classification. The relevant pattern
in prior work is:

1. Represent each problem instance by interpretable meta-features.
2. Estimate which algorithm or method family will improve on a default.
3. Prefer performance estimation, ranking, or pairwise decisions when direct
   best-method classification is unstable.
4. Use compact time-series statistics, frequency properties, entropy,
   volatility, autocorrelation, and distributional descriptors as features.

This motivated changing the router target from "predict the best config" to
"predict whether a specific activation regime or placement should be enabled."

## Method

Implemented script:

- `lee_ocil/scripts/phase4_pairwise_factor_router.py`

Output root:

- `phase4_remote_results/analysis_current/factors_router/pairwise_uplift/`

The method constructs pairwise uplift rows:

```text
uplift(action, comparator) = (MSE(comparator) - MSE(action)) / MSE(comparator)
```

Positive uplift means the candidate action improves on the comparator.

The router learns single-factor threshold rules from training folds:

```text
recommend action if factor <= threshold
recommend action if factor >= threshold
otherwise abstain
```

The policy value is:

```text
policy_uplift = uplift if rule recommends action else 0
```

This directly measures whether the factor can safely decide when to enable a
regime or placement. It avoids using held-out best config labels as features.

## Action Pairs

Regime pairs:

- `regime_best_bounded_vs_default`
- `regime_tanh_vs_default`
- `regime_tanh_sin_vs_default`
- `regime_softsign_vs_default`
- `regime_best_unbounded_vs_default`
- `regime_swish_vs_default`
- `regime_relu_vs_default`
- `regime_tanh_sin_vs_tanh`
- `regime_softsign_vs_tanh`

Placement pairs:

- `placement_output_tanh_matched`
- `placement_output_tanh_vs_best_linear`
- `placement_encoder_tanh_vs_default`
- `placement_decoder_tanh_vs_default`
- `placement_all_tanh_vs_best_split`

## Key Findings

### Regime

Bounded activations remain a strong static policy:

- `regime_best_bounded_vs_default`: mean uplift `0.0931`,
  positive rate `0.6491`.
- `regime_tanh_vs_default`: mean uplift `0.0779`,
  positive rate `0.6316`.
- `regime_tanh_sin_vs_default`: mean uplift `0.0767`,
  positive rate `0.6140`.
- `regime_softsign_vs_default`: mean uplift `0.0722`,
  positive rate `0.5789`.

Factor routing does not clearly beat always trying bounded regimes. Topology
features such as `d_model` and `pred_to_seq_ratio` are stronger than pure data
factors for deciding bounded-vs-default across architectures. Data factors
such as `input_volatility`, `input_turbulence_score`, and `input_cv` identify
many positive bounded-regime cases, but the policy is usually below the
always-action uplift because bounded regimes are already often good.

Interpretation:

- Regime is currently best framed as architecture/topology-conditioned static
  policy, with data factors used for confidence rather than hard selection.

### Placement

Placement has the clearest factor-conditioned signal.

#### All-tanh vs split encoder/decoder

For `placement_all_tanh_vs_best_split`, frequency/time-frequency factors
produce the strongest router result:

- Leave-dataset policy uplift: `0.0668`.
- Always-action uplift: `0.0554`.
- Excess over always-action: `+0.0114`.
- Recommend rate: `0.5556`.
- Precision if recommended: `0.8500`.
- Mean uplift if recommended: `0.1202`.
- Main selected factors:
  - `router_timefreq_rolling_spectral_entropy_std`
  - `router_timefreq_burstiness_index`

Representative rule:

```text
if rolling spectral entropy std is low, prefer all-tanh over the best split placement.
```

This is the first strong evidence that factor buckets can judge placement,
not merely activation regime.

#### Encoder tanh vs default

For `placement_encoder_tanh_vs_default`:

- Always-action uplift: `0.0573`.
- Frequency policy uplift: `0.0550`.
- Precision if recommended: `0.7188`.
- Mean uplift if recommended: `0.0619`.
- Main selected factors:
  - `router_timefreq_dominant_frequency_drift`
  - `router_timefreq_phase_concentration`

Interpretation:

- Encoder tanh is broadly useful in Informer, but the factor rule mostly acts
  as a high-confidence filter rather than improving over always enabling it.

#### Decoder tanh vs default

For `placement_decoder_tanh_vs_default`:

- Always-action uplift: `0.0242`.
- Frequency policy uplift: `0.0222`.
- Precision if recommended: `0.6000`.
- Mean uplift if recommended: `0.0266`.
- Main selected factors:
  - `router_timefreq_phase_concentration`
  - `router_timefreq_dominant_frequency_drift`

Interpretation:

- Decoder tanh is weaker than encoder tanh and less cleanly routed.

#### Output tanh

Output tanh is globally harmful in the current combined matrix:

- `placement_output_tanh_matched`: mean uplift `-0.4248`,
  positive rate `0.1778`.
- `placement_output_tanh_vs_best_linear`: mean uplift `-0.4753`,
  positive rate `0.1556`.

The frequency rule can identify a tiny positive subset:

- `placement_output_tanh_vs_best_linear`, leave-dataset:
  - policy uplift `0.0017`
  - recommend rate `0.0667`
  - precision `0.6667`
  - selected factor: `router_freq_dominant_period_to_horizon_ratio`

Interpretation:

- Output tanh should be an abstain-heavy placement action. It should not be a
  default placement and needs more iTransformer-specific confirmation before it
  becomes a paper claim.

## Current Answer to the Research Question

Can factors judge placement and regime?

Yes, partially:

- Placement: yes, especially `all-tanh vs split placement` in Informer, where
  time-frequency stability/burstiness factors improve over always-action and
  reach high recommendation precision.
- Regime: weaker. Bounded regimes are often good by static policy, so data
  factors mostly provide confidence, not a better decision rule.
- Output placement: mostly no. Existing factors can identify rare safe cases,
  but the action is globally harmful and should be gated very conservatively.

## Recommended Next Experimental Matrix

The next experiments should validate the pairwise action rules rather than
rerunning a broad config grid.

1. Informer placement confirmation:
   - Focus on cells where the rule recommends all-tanh vs split and where it
     abstains.
   - Compare `tanh_all`, `enc_tanh_dec_gelu`, `enc_gelu_dec_tanh`,
     `tanh_sin001_all`, and `gelu_all`.
   - Use 3 seeds if not already complete.

2. iTransformer output-tanh confirmation:
   - Select a small number of cells with low dominant-period-to-horizon ratio
     and matched cells with high ratio.
   - Compare `gelu_ffn_linear`, `gelu_ffn_outtanh`, `tanh_ffn_linear`,
     `tanh_ffn_outtanh`, `tanh_sin001_ffn_linear`,
     `tanh_sin001_ffn_outtanh`.
   - Do not claim output-tanh routing until this pairwise factor holds.

3. PatchTST negative-transfer confirmation:
   - Confirm that output-tanh remains harmful in both recommend and abstain
     buckets.
   - Keep PatchTST as an architecture-specific counterexample in the story.

## Paper Claim Boundary

Safe claim:

```text
Factor buckets can identify high-confidence placement regimes, especially
whether full bounded activation should replace split encoder/decoder bounded
placement under stable time-frequency structure.
```

Unsafe claim:

```text
The current router can generally select activation function and placement
zero-shot across Transformer architectures.
```

