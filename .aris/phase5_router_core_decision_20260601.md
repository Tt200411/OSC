# Phase-5 Router-Core Result-to-Claim Decision

Generated: 2026-06-01

## Verdict

`claim_supported = partial`

The current evidence does **not** support a strong zero-shot factor-router claim. It does support a narrower **oracle-supplemented / calibrated low-cost activation selector** route:

- Pure zero-shot rules trained only on the old Informer full grid fail on Weather / Exchange / ILI.
- Pairwise rule calibration with leave-new-dataset / leave-new-cell splits does not recover stable gains versus GELU.
- A seed-2024 small oracle matrix used as calibration, then evaluated on held-out seeds 2025/2026, gives a weak positive route: mean gain vs GELU is positive overall and regret vs held-out oracle is low, but performance is not uniformly positive across datasets.

## Evidence Completed

- New dataset matrix: `Weather`, `Exchange`, `ILI`
- Configs: `gelu_all`, `tanh_sin001_all`, `tanh_all`, `softsign_all`, `enc_tanh_dec_gelu`, `enc_gelu_dec_tanh`, `tanh_all_outtanh`
- Seeds: `2024`, `2025`, `2026`
- Completed seed-runs: `189/189`
- Missing seed-runs: `0`
- Remote status: no stuck / failed / OOM / NaN alerts in queue summaries

Primary artifacts:

- `.aris/phase5_router_oracle_eval_20260601_all3seeds_zero/`
- `.aris/phase5_router_oracle_eval_20260601_all3seeds_taskctx/`
- `.aris/phase5_calibrated_router_20260601_all3seeds_taskctx/`
- `.aris/phase5_oracle_supplement_selection_20260601_all3seeds/`

## Three-Seed Oracle Pattern

The oracle itself confirms that activation and placement remain dataset/horizon-sensitive:

| dataset | pred_len | oracle_config | oracle_gain_vs_gelu | best_static |
| --- | ---: | --- | ---: | --- |
| Exchange | 96 | `enc_gelu_dec_tanh` | 3.46% | `tanh_sin001_all` |
| Exchange | 192 | `enc_gelu_dec_tanh` | 1.44% | `softsign_all` |
| Exchange | 336 | `enc_gelu_dec_tanh` | 0.76% | `softsign_all` |
| ILI | 24 | `tanh_sin001_all` | 2.06% | `tanh_sin001_all` |
| ILI | 36 | `enc_tanh_dec_gelu` | 4.01% | `tanh_sin001_all` |
| ILI | 48 | `tanh_all` | 3.41% | `tanh_all` |
| Weather | 96 | `tanh_sin001_all` | 4.87% | `tanh_sin001_all` |
| Weather | 192 | `enc_gelu_dec_tanh` | 0.26% | `tanh_sin001_all` |
| Weather | 336 | `enc_gelu_dec_tanh` | 1.36% | `tanh_sin001_all` |

This is positive for the activation/placement search problem, especially for placement. It is not yet positive for a purely zero-shot selector.

## Router Results

### Pure Zero-Shot

Old Informer full-grid rules choose mostly `tanh_sin001_all`.

| metric | value |
| --- | ---: |
| mean gain vs GELU | -11.12% |
| positive vs GELU rate | 4/9 |
| mean regret vs oracle | 13.70% |
| config hit rate | 2/9 |
| abstain rate | 0 |

Adding task/topology context did not change this result.

Decision: zero-shot cross-dataset router is **not supported**.

### Pairwise Calibrated Splits

Using the new oracle aggregate in leave-new-cell / leave-new-dataset / leave-new-horizon splits still fails versus GELU. The best-looking split is `supplement_leave_new_horizon` with frequency features:

| metric | value |
| --- | ---: |
| mean gain vs GELU | -6.16% |
| mean gain vs static | +3.88% |
| config hit rate | 4/9 |
| action hit rate | 9/9 |
| regret vs oracle | 8.67% |

Decision: pairwise factor rules can learn the action family better than exact config, but still cannot be claimed as a deployable cross-dataset router.

### Oracle-Supplement Selection

Use seed-2024 oracle to choose the config per dataset-horizon, then evaluate that selection on held-out seeds 2025/2026.

| scope | gain vs GELU | gain vs train static | regret vs held-out oracle | positive vs GELU |
| --- | ---: | ---: | ---: | ---: |
| Exchange | +1.19% | +13.60% | 0.69% | 2/3 |
| ILI | +2.27% | -0.79% | 1.01% | 3/3 |
| Weather | -1.40% | +15.88% | 3.48% | 0/3 |
| ALL | +0.69% | +9.56% | 1.73% | 5/9 |

Decision: oracle-supplement has a **weak but real positive signal**. It is strong against the old static baseline, slightly positive versus GELU overall, and close to held-out oracle. It is not uniformly stable across datasets.

## Supported Claim

A conservative supported claim is:

> Activation regime and placement are dataset/horizon-sensitive in Transformer-style Informer. A small oracle calibration matrix can be used as a low-cost activation/placement selector that improves over the old static activation baseline and gives modest overall held-out gains versus GELU, but pure zero-shot cross-dataset routing is not yet reliable.

## Unsupported Claim

Do not claim:

- A zero-shot router trained only on old datasets reliably improves Weather / Exchange / ILI.
- The current pairwise factor rules are a deployable cross-dataset selector.
- `tanh_sin001_all` is a universal strong transfer config.
- Router beats GELU consistently on every new dataset/horizon.

## Route

`CALIBRATE`, not `GO`.

The router can remain the paper method only if the first version frames it as:

- **factor-guided calibrated activation/placement selection**, or
- **oracle-supplemented low-cost router**, where a small new-dataset oracle matrix is part of the method.

If the paper must be strictly zero-shot, the current result forces a pivot: the router becomes a diagnostic / search-prior rather than the core algorithm.

## Next Implementation Direction

1. Add an explicit abstain/OOD gate so old-grid rules do not force `tanh_sin001_all` on Weather long horizons and Exchange.
2. Treat horizon as a first-class task feature, but keep claims honest: task context alone did not fix zero-shot.
3. Reframe the router target from exact config prediction to action-family recommendation plus calibrated fallback.
4. For paper v1, use the three-seed cross-dataset matrix as the router stress test and present oracle-supplement selection as the positive calibrated path.
