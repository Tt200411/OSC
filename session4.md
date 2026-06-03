# Session 4: Phase-3 Regime, Placement, and Baseline Controls

Date: 2026-05-17

## Objective

Implement the next paper-stage plan without waiting for slow Lee queues:

- run additional exact-protocol controls on cloud RTX 4090s;
- add ReLU and Swish as non-bounded baselines;
- verify placement effects on ETT and Solar96;
- expand Solar96 regime/factor diagnostics toward finance-style factors;
- update the paper narrative conservatively.

## Remote execution status

Four Phase-3 batches were launched on remote 4090 hosts and pulled back locally:

| Host | Workload | Expected rows | Status |
|---|---|---:|---|
| `10.21.53.62` | ETTh1 168/336, ReLU/Swish, seeds 2024-2026 | 12 | complete |
| `10.21.53.113` | ETTh2 168/336, ReLU/Swish, seeds 2024-2026 | 12 | complete |
| `10.21.53.142` | Solar1/96, ReLU/Swish, seeds 2024-2026 | 6 | complete |
| `10.21.53.82` | Solar5/96, ReLU/Swish, seeds 2024-2026 | 6 | complete |

Pulled summaries:

- `phase3_remote_results/10.21.53.62/results/summary.csv`
- `phase3_remote_results/10.21.53.113/results/summary.csv`
- `phase3_remote_results/10.21.53.142/results/summary.csv`
- `phase3_remote_results/10.21.53.82/results/summary.csv`

Combined analysis output:

- `phase3_remote_results/analysis_current/phase2_protocol_raw_dedup.csv`
- `phase3_remote_results/analysis_current/phase2_protocol_aggregate.csv`
- `phase3_remote_results/analysis_current/phase2_informer_table2_compare.csv`

Combined Phase-2/3 filtering produced 569 raw phase-tagged rows, 259 deduplicated protocol rows, and 97 aggregate rows. Main claims still use batch size 32, six epochs, and three seeds.

## Main Phase-3 control result

ReLU and Swish do not overturn the bounded/tanh-family story.

| Dataset | H | GELU | ReLU | Swish | Best bounded/tanh-family | Gain vs GELU |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 168 | 1.046 | 1.063 | 1.022 | tanh 0.908 | +13.2% |
| ETTh1 | 336 | 1.355 | 1.378 | 1.296 | softsign 1.219 | +10.0% |
| ETTh2 | 168 | 5.142 | 6.726 | 4.004 | tanh_sin 2.976 | +42.1% |
| ETTh2 | 336 | 3.928 | 4.219 | 3.074 | tanh_sin 2.172 | +44.7% |
| Solar1 | 96 | 0.230 | 0.236 | 0.226 | encoder-tanh 0.224 | +2.5% |
| Solar5 | 96 | 0.195 | 0.198 | 0.194 | encoder-tanh 0.180 | +7.6% |

Interpretation:

- ReLU is mostly worse than GELU.
- Swish is a strong smooth baseline on ETTh2 and Solar1, but remains weaker than bounded/tanh-family choices.
- Solar96 confirms the placement story: encoder-side tanh is stronger than decoder-only or output tanh.

## Placement result

The current strongest placement claims are:

- ETTh1: hidden tanh is useful; output tanh is worse than linear output.
- ETTh2: output tanh can help, especially at horizon 336, but this is dataset/horizon specific.
- Solar1/96 and Solar5/96: encoder-side tanh is best among tested placements.
- Solar96 output tanh is harmful and should not be merged into the hidden-activation mechanism.

## Factor/oracle diagnostics

New factor generation adds finance-style input-window factors:

- realized volatility, volatility shock, upside/downside volatility;
- momentum, trend consistency, reversal rate;
- lag autocorrelation, mean-reversion strength;
- jump intensity, tail ratio, drawdown/drawup, direction entropy.

Solar96 target-channel static ranking:

| Dataset | Best | MSE | Next | MSE | GELU |
|---|---|---:|---|---:|---:|
| Solar1/96 | encoder-tanh | 0.177558 | full tanh | 0.177967 | 0.192832 |
| Solar5/96 | encoder-tanh | 0.250646 | tanh_sin | 0.255040 | 0.281201 |

Oracle conclusion:

- Same-split factor oracles show small diagnostic gains.
- Leave-one-seed-out gains are positive only on Solar1/96 for max absolute change (+1.57%), volatility shock (+1.26%), and volatility (+0.92%).
- Solar5/96 has no positive LOSO factor oracle.

This supports a conditional-regime story but not a deployable activation router yet.

## Paper update

Updated the current paper to reflect Phase-3:

- title changed toward placement/regime diagnostics;
- abstract and introduction now mention ReLU/Swish controls;
- method now includes ReLU and Swish as non-bounded controls and defines regime diagnostics;
- main ETT table includes ReLU and Swish;
- placement table includes ReLU/Swish and Solar96 placement;
- new Solar96 oracle diagnostic table added;
- conclusion narrows the claim boundary.

## Current claim boundary

Supported:

1. Activation-only changes matter under matched Informer protocol.
2. Bounded/tanh-family hidden activations beat same-environment GELU in the four ETT long-horizon cells.
3. ReLU/Swish controls do not explain away the bounded/tanh-family advantage.
4. Placement matters: encoder/hidden boundedness and output bounding are different mechanisms.
5. Finance-style factors provide regime diagnostics, but only weak leave-one-seed-out routing evidence so far.

Not supported:

1. Tanh is universally best.
2. High volatility always prefers tanh or oscillatory tanh.
3. The current runs reproduce and surpass Informer Table 2.
4. Lee is established as a main exact-protocol replacement.
5. A factor-based activation router is ready as a main-method claim.
