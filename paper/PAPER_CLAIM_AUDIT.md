# Paper Claim Audit Report

**Date**: 2026-05-16
**Auditor**: GPT-5.5 xhigh fresh zero-context thread
**Overall Verdict**: WARN

## Summary

All audited quantitative values matched raw CSV evidence exactly or by rounding. The audit found no numeric mismatch, config mismatch, or aggregation mismatch after the wording fixes. The verdict remains WARN because the paper explicitly reports Phase-1 descriptive signals, not final inferential statistics.

## Counts From Fresh Audit

- exact_match: 277
- rounding_ok: 229
- ambiguous_mapping: 1, fixed in `sections/3_method.tex` by adding `f_rand`
- missing_evidence: 1, fixed in the audit manifest by using `lee_ocil/scripts/run_phase1_matrix.sh`
- scope_overclaim: 1, fixed in `figures/gen_phase1_figures.py` and regenerated `paper/figures/TABLE_bounded_vs_gelu.tex`
- number_mismatch: 0
- config_mismatch: 0
- aggregation_mismatch: 0
- unsupported_claim: 0

## Residual Risk

The paper is numerically faithful to the synchronized CSV snapshot, but the snapshot is not a frozen inferential dataset. This is disclosed in the abstract, experiments, conclusion, and appendix.
