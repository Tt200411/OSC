# Phase-4 Factor-Router Claim Summary

Date: 2026-05-23

## Scope

This note summarizes the current evidence for a zero-shot activation
regime/placement router using:

- Informer full grid: 36 dataset-horizon cells, 9 configs, 3 seeds.
- PatchTST transfer probes: 12 cells.
- iTransformer stage-1 FFN probes: 6 cells.
- iTransformer placement probes: 3 cells.

The router sample size is therefore 57 model-cells, not 972 seed-runs.

## Main Verdict

Current evidence does not support a deployable zero-shot router that directly
selects the exact activation regime and placement.

The evidence does support a weaker and useful claim:

1. Activation behavior is architecture-dependent.
2. Informer has a strong static bounded-activation policy, with
   `tanh_sin001_all` as the best mean-rank static policy.
3. PatchTST mostly prefers `gelu_ffn_linear`; output tanh is a negative
   transfer in the tested placement cells.
4. iTransformer shows early positive placement signal, especially for
   output-tanh on some horizons, but the evidence is still small.
5. Data/time-frequency factors contain weak-to-moderate diagnostic signal, but
   they do not yet provide a robust deployable exact-config router.

## Key Numbers

### Informer input-only router

- Static baseline by mean rank: `tanh_sin001_all`.
- Static mean gain vs GELU: 0.1237.
- Oracle mean gain vs GELU: 0.1597.
- Static mean regret vs oracle: 0.0472.
- LOCO nested single-factor router mean gain vs static: 0.00035.
- LOCO positive-vs-static rate: 0.2778.
- Verdict: no-go for an Informer-only deployable router.

### Cross-architecture action-class upper bound

This setting predicts an action class and then evaluates the best config inside
that action class in the held-out cell. It is useful as an upper bound, but is
not directly deployable.

- Leave-cell mean gain vs static: 0.0198.
- Leave-cell positive-vs-static rate: 0.4912.
- Leave-cell mean regret vs oracle: 0.00775.
- Verdict: partial-go as an action-class upper bound only.

### Cross-architecture deployable exact-config router

This setting predicts a concrete config from the training fold and evaluates
that config in the held-out cell.

- Leave-cell mean gain vs static: 0.00233.
- Leave-cell positive-vs-static rate: 0.2982.
- Leave-cell config hit rate: 0.3684.
- Leave-dataset mean gain vs static: 0.00225.
- Leave-dataset positive-vs-static rate: 0.2807.
- Leave-dataset config hit rate: 0.2982.
- Leave-model-scope and leave-family fail because unseen model scopes fall back
  to the baseline config.
- Verdict: no-go for a deployable exact-config zero-shot router.

## Feature Ablation

Deployable config-level ablation shows that data factors are not yet stronger
than simple architecture/topology or scope-level rules.

| feature set | leave-cell gain vs static | leave-dataset gain vs static |
| --- | ---: | ---: |
| constant scope rule | -0.00086 | 0.00308 |
| data only | -0.00060 | 0.00258 |
| frequency only | -0.00128 | 0.00187 |
| statistical only | -0.00060 | 0.00270 |
| topology only | 0.00233 | 0.00308 |
| model ID only | -0.00086 | 0.00308 |
| all features | 0.00233 | 0.00308 |

Top cross-architecture correlations are dominated by topology/protocol features:
`n_heads`, `d_model`, `train_epochs`, `d_ff`, `scope_config_count`, `d_layers`.

The strongest data factors are weaker:

- `router_freq_seasonal_peak_strength`: Spearman -0.383 with oracle gain.
- `input_daily_cycle_energy_ratio`: Spearman -0.343 with oracle gain.
- `router_multivar_correlation_condition_number`: Spearman 0.321 with oracle
  gain.
- `input_spectral_entropy`: Spearman 0.287 with oracle gain.

## Additional Two-Stage Scope-Local Attempt

After the main model-aware no-go result, we also tested a more realistic
two-stage hypothesis:

1. Gate by model scope/architecture.
2. Within each seen scope, learn a data-factor refinement rule.
3. Select both the scope static comparator and the dynamic rule from the
   training fold.

This is stricter than using the all-data static config and closer to a
deployment setting over already-seen model scopes.

### Exact-config refinement

| feature set | leave-cell gain vs train-scope static | leave-dataset gain vs train-scope static |
| --- | ---: | ---: |
| data/statistical factors | 0.0135 | 0.0020 |
| frequency-only factors | -0.0063 | -0.0169 |

The data/statistical refinement signal is positive in leave-cell, mainly from
Informer (+0.0213 within Informer), but weak in leave-dataset and not yet stable
for PatchTST/iTransformer because those scopes have few cells.

### Action-level refinement

Action-level refinement does not improve the story:

| feature set | leave-cell gain vs train-scope static | leave-dataset gain vs train-scope static |
| --- | ---: | ---: |
| frequency-only action rule | 0.0006 | -0.0005 |
| data/statistical action rule | -0.0003 | -0.0042 |

This suggests that the current useful signal, when present, is not simply
"choose bounded vs GELU vs output-tanh"; it is closer to exact-config
scope-local refinement over seen architectures.

## Claim Boundary for the Paper

Do not claim that the current method can zero-shot select the exact activation
function and placement.

A defensible current framing is:

- Static bounded activations transfer strongly within Informer.
- Activation transfer is not architecture-invariant.
- Data/time-frequency factors are useful diagnostics and may help define
  candidate regimes, but architecture/topology is currently the stronger
  router signal.
- A practical router should be framed as a two-stage hypothesis:
  architecture/topology gate first, then data-factor refinement within the
  architecture.
- The best current router evidence is a weak two-stage, seen-scope refinement
  result, not cross-architecture zero-shot generalization.

## Minimal Experiments Needed

1. Expand iTransformer placement probes to 3 seeds and more cells, especially
   ETTh2/96, ETTh2/336, ETTh2/720, ETTm2/96, ETTm2/336, ETTm2/720, and at
   least one ETTh1/ETTm1 contrast.
2. Confirm PatchTST negative transfer with 3 seeds for the current transfer
   cells, focusing on GELU-linear vs bounded-linear, not output-tanh.
3. Re-run router evaluation after balancing model-cell counts. Current
   cross-architecture evidence is dominated by 36 Informer cells.
4. If a zero-shot router remains the paper target, evaluate a two-stage router:
   model/topology gate -> within-architecture data-factor rule.
5. Treat same-split/action-class oracle results as diagnostic upper bounds only.
