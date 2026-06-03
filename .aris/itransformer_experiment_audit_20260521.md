# iTransformer Experiment Audit Report

**Date**: 2026-05-21
**Scope**: iTransformer activation-transfer quick validation on ETTh2/ETTm2, horizons 96/336/720, seed 2024.
**Auditor**: local executor audit; external reviewer not spawned because this session did not explicitly authorize sub-agents.

## Overall Verdict: WARN

Integrity status is `warn`, not because the files are inconsistent, but because the scope is intentionally small: one seed, two datasets, and FFN activation only. The result is suitable as a fast transfer probe, not as a final generalization claim.

## Coverage

- Expected seed-runs: 24
- Reusable seed-runs: 2
- Newly launched seed-runs: 22
- Remote queue terminal status: 10.21.53.113: {'completed': 6}, 10.21.53.142: {'completed': 5}, 10.21.53.162: {'completed': 5}, 10.21.53.82: {'completed': 6}
- Finite deduplicated rows: 24/24
- Duplicate dataset/horizon/config/seed rows after dedup: 0
- Artifact coverage in final result root: config=24/24, metrics=24/24, pred=24/24, true=24/24

## Checks

### A. Ground Truth Provenance: PASS

ETT data are loaded from CSV files in `iTransformer/data_provider/data_loader.py:44` and `iTransformer/data_provider/data_loader.py:132`; test targets come from `batch_y` in `iTransformer/experiments/exp_long_term_forecasting.py:276-308`. Predictions and true arrays are appended separately at `iTransformer/experiments/exp_long_term_forecasting.py:314-318`.

### B. Score Normalization: PASS

MSE/MAE are raw error metrics from `iTransformer/utils/metrics.py:14-40`; no metric is normalized by prediction statistics. The only relative gains are post-hoc paired comparisons against GELU in `iTransformer/scripts/itransformer_aggregate.py:70-72`.

### C. Result File Existence: PASS

All 24 deduplicated rows have `config.json`, `metrics.npy`, `pred.npy`, and `true.npy` in `itransformer_remote_results/stage1_actfix3_20260521/`.

### D. Completion And Queue Integrity: PASS

All 22 newly launched jobs reached `completed` in queue state; the remaining 2 rows are documented smoke reuse under the same protocol.

### E. Scope Assessment: WARN

This is one seed (`2024`) and two ETT datasets only. It can support a quick architecture-transfer stress-test statement, but not a robust cross-architecture conclusion.

### F. Evaluation Type: real_gt

Evaluation compares model predictions with dataset-provided held-out `batch_y`; no synthetic or model-derived ground truth is used.

## Action Items

- Do not claim iTransformer generalization from this run alone.
- If the paper needs stronger architecture-transfer evidence, add seeds 2025/2026 for the same 24 cells or narrow the claim to a single-seed pilot.
- Placement/output-tanh was intentionally not tested here, so no iTransformer placement claim should be made.
