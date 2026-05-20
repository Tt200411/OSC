# Result-To-Claim Prompt

Judge whether the completed Phase-4 full-grid Informer activation experiments support the intended claim.

## Intended Claim

Activation-only changes can improve Informer forecasting under fixed architecture and training protocol, and the useful activation depends on dataset, horizon, and placement.

## Evidence To Judge

- Expected seed-runs: 972.
- Reusable seed-runs: 972.
- Missing seed-runs: 0.
- Aggregate rows: 324.
- Complete dataset-horizon-config cells: 324/324.
- Dataset-horizon cells: 36.
- Non-GELU best cells: 34/36.
- GELU-best cells: Solar3/24 and Solar3/96.
- Paired non-GELU vs GELU seed comparisons: 864/864.
- LOSO factor-oracle rows: 84; positive rows: 7, all on Solar1/96.

## Known Caveats

- Same-protocol evidence only.
- Not an exact reproduction of original Informer Table 2.
- Same-split factor oracles are diagnostic upper bounds.
- Lee/small-batch diagnostics are excluded from main claims.
