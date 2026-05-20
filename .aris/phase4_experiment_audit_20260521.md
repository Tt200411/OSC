# Phase-4 Experiment Audit Report

**Date**: 2026-05-21  
**Auditor**: local Codex audit, pending external reviewer  
**Scope**: Phase-4 full-grid Informer activation experiments  
**Overall Verdict**: PASS with scope caveats

## Summary

The Phase-4 result set is complete for the planned main matrix: 972/972 expected seed-runs are reusable, 0 are missing, and all 324 dataset-horizon-config cells have three seeds under the canonical batch-size-32, six-epoch protocol. The audit found no duplicate canonical rows, no wrong batch/epoch rows in the main aggregate, no non-finite MSE/MAE values, and no missing GELU pairings for paired comparisons.

The claim must remain same-environment and protocol-local. The evidence supports dataset- and horizon-sensitive activation choice under the matched Informer protocol. It does not support a universal activation, a deployed factor router, or a claim of reproducing and surpassing original Informer Table 2 values.

## Checks

### A. Ground Truth Provenance: PASS

Ground truth comes from dataset CSV splits, not from model outputs.

- Dataset defaults map ETT datasets to `OT` and Solar sites to `Power`: `lee_ocil/phase4_informer.py:18` and `lee_ocil/phase4_informer.py:25`.
- Phase-4 data loading selects ETT hour/minute loaders and custom Solar loaders, then returns dataset batches from CSV-backed datasets: `lee_ocil/phase4_informer.py:68`.
- Dataset loaders read CSV files and construct `data_y` from the scaled dataset values, not predictions: `lee_ocil/data/data_loader.py:45`, `lee_ocil/data/data_loader.py:131`, and `lee_ocil/data/data_loader.py:218`.
- Test evaluation stores `preds` and `trues` separately from `_process_one_batch`: `lee_ocil/exp/exp_informer.py:239`.

### B. Score Normalization: PASS

Primary metrics are raw MSE/MAE against true targets.

- Metric functions compute raw `np.mean((pred-true)**2)` and `np.mean(np.abs(pred-true))`: `lee_ocil/utils/metrics.py:11`.
- Test writes MSE/MAE from `metric(preds, trues)` to arrays and summary rows: `lee_ocil/exp/exp_informer.py:266`.
- Relative improvement divides by matched GELU MSE, not by prediction statistics: `lee_ocil/scripts/phase4_aggregate.py:14`.

### C. Result File Existence And Matrix Integrity: PASS

Local integrity check:

- Expected matrix rows: 972.
- Reusable rows: 972.
- Missing rows: 0.
- Data integrity: 12/12 CSV files passed, target NaN sum 0, total NaN sum 0.
- Raw deduplicated rows: 972.
- Aggregate rows: 324.
- Complete config cells: 324/324.
- Paired-vs-GELU rows: 864/864 expected non-GELU seed pairs.

Implementation evidence:

- Canonical expected matrix enumerates 12 datasets, configured horizons, nine configs, and seeds 2024/2025/2026: `lee_ocil/scripts/phase4_common.py:20`, `lee_ocil/scripts/phase4_common.py:80`, and `lee_ocil/scripts/phase4_common.py:264`.
- Reusable-run filtering requires config readability, finite MSE/MAE, canonical batch size, canonical epoch count, matching attention/embed fields, and excludes batch-adjusted rows: `lee_ocil/scripts/phase4_common.py:502`.
- Completion matrix requires all expected seed keys per dataset-horizon-config cell: `lee_ocil/scripts/phase4_aggregate.py:97`.

### D. Duplicate Rows And Protocol Mismatch: PASS

Canonical duplicate counts:

- Expected duplicate keys: 0.
- Reusable duplicate keys: 0.
- Raw deduplicated duplicate keys: 0.
- Wrong batch rows in main raw file: 0.
- Wrong epoch rows in main raw file: 0.
- Non-finite MSE rows: 0.
- Non-finite MAE rows: 0.

The deduplication key includes dataset, horizon, sequence length, label length, layer counts, factor, batch size, epochs, activation signature, and seed: `lee_ocil/scripts/phase4_common.py:201`.

### E. Scope Assessment: PASS with caveats

Actual scope is broad enough for the paper's same-protocol claim:

- 36 dataset-horizon cells.
- 9 activation configs.
- 3 seeds per config.
- 972 seed-runs.

Scope caveats:

- This is not an exact reproduction of original Informer Table 2.
- Factor/oracle evidence is diagnostic. Same-split oracles are upper bounds; leave-one-seed-out evidence is weak and positive only for Solar1/96.
- Batch-adjusted Lee and earlier small-batch diagnostics are excluded from main claims.

### F. Evaluation Type: real_gt

The main evaluation uses real dataset targets from ETT and Solar CSV files. Factor analyses use post hoc regime annotations and saved predictions, but the oracle metrics are still computed against saved true arrays. Factor-router claims are not made.

## Claim Impact

- Supported: activation choice is dataset- and horizon-sensitive under the matched Phase-4 Informer protocol.
- Supported: bounded/tanh-family configs are strong static choices by mean rank.
- Dataset-specific: Solar3 remains GELU-best; Solar gains are site-specific.
- Not supported: universal bounded/oscillatory activation win.
- Not supported: deployed factor-conditioned activation router.
- Not supported: surpassing the original Informer paper.

## External Review Status

The usual cross-model reviewer step was not run in this turn because agent delegation is unavailable under the active tool policy. This report records local, file-backed audit evidence and should be treated as pending external reviewer confirmation.
