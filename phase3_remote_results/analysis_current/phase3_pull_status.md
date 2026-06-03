# Phase 3 Pull/Aggregation Status

Date: 2026-05-17 Asia/Shanghai  
Role: Agent C results pull + aggregation  
Scope: read remote summaries/logs if accessible; write local `phase3_remote_results` / `.aris` outputs only.

## Pull Status

Batch SSH failed for all requested hosts with `Permission denied (publickey,password)`. No password was written to disk and no remote jobs were launched.

| Host | Status | summary.csv | logs |
|---|---|---:|---:|
| 10.21.53.62 | SSH auth blocked | no | no |
| 10.21.53.113 | SSH auth blocked | no | no |
| 10.21.53.142 | SSH auth blocked | no | no |
| 10.21.53.82 | SSH auth blocked | no | no |

## Compact Summary

- New `phase3_regime_place` row count: 0
- Completed new configs: 0
- Combined aggregate output: `phase3_remote_results/analysis_current/phase3_combined_protocol_aggregate.csv` (phase2-only copy because no new phase3 rows were available)
- New-row snapshot: `phase3_remote_results/analysis_current/phase3_regime_place_new_rows.csv`
- Host status CSV: `phase3_remote_results/analysis_current/phase3_pull_host_status.csv`

## Preliminary Mean/Std Table

No new Phase-3 rows were available. Current reference is the existing Phase-2 aggregate preview:

| dataset | pred_len | activation | encoder_activation | decoder_activation | output_activation | mse_mean | mse_std | mae_mean | mae_std | seed_count | run_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ETTh1 | 168 | tanh | tanh | tanh | linear | 0.911571 | 0.130753 | 0.743789 | 0.0534551 | 3 | 3 |
| ETTh1 | 168 | tanh_sin | tanh_sin | tanh_sin | linear | 0.920825 | 0.137783 | 0.74885 | 0.0580847 | 3 | 3 |
| ETTh1 | 168 | softsign | softsign | softsign | linear | 0.932708 | 0.146642 | 0.748641 | 0.0645031 | 3 | 3 |
| ETTh1 | 168 | tanh | tanh | tanh | linear | 0.946928 | 0.0569576 | 0.774032 | 0.0393494 | 3 | 3 |
| ETTh1 | 168 | softsign | softsign | softsign | linear | 0.960911 | 0.0855003 | 0.7908 | 0.0445078 | 3 | 3 |
| ETTh1 | 168 | lee | lee | lee | linear | 0.978981 | 0.0103885 | 0.792599 | 0.00465757 | 3 | 3 |
| ETTh1 | 168 | gelu | tanh | gelu | linear | 1.03587 | 0.097441 | 0.80305 | 0.0260519 | 3 | 3 |
| ETTh1 | 168 | gelu | gelu | gelu | linear | 1.03921 | 0.039915 | 0.830425 | 0.0189335 | 3 | 3 |
| ETTh1 | 168 | gelu | gelu | gelu | linear | 1.04711 | 0.0209829 | 0.825594 | 0.0136836 | 3 | 3 |
| ETTh1 | 168 | gelu | gelu | tanh | linear | 1.05582 | 0.0248576 | 0.822808 | 0.010206 | 3 | 3 |
| ETTh1 | 168 | lee | lee | lee | linear | 1.09084 | 0.0859086 | 0.817621 | 0.0377295 | 3 | 3 |
| ETTh1 | 168 | tanh | tanh | tanh | tanh | 1.10571 | 0.0140117 | 0.848319 | 0.0123115 | 3 | 3 |

## Missing Cells / Blockers

- Missing all Phase-3 cells from the four requested hosts because SSH authentication is unavailable in this session.
- Missing `~/project/osc_informer/lee_ocil/results/summary.csv` pulls for all requested hosts.
- Missing `~/project/osc_informer/lee_ocil/logs/` pulls for all requested hosts.
- Missing actual `phase3_regime_place` completion assessment until at least one new summary is pulled.

## Exact Retry Commands

After an interactive auth method or agent-loaded SSH key is available:

```bash
mkdir -p phase3_remote_results/{10.21.53.62,10.21.53.113,10.21.53.142,10.21.53.82}
for h in 10.21.53.62 10.21.53.113 10.21.53.142 10.21.53.82; do
  mkdir -p phase3_remote_results/$h/results phase3_remote_results/$h/logs
  rsync -av --ignore-missing-args "$h:~/project/osc_informer/lee_ocil/results/summary.csv" "phase3_remote_results/$h/results/"
  rsync -av --ignore-missing-args "$h:~/project/osc_informer/lee_ocil/logs/" "phase3_remote_results/$h/logs/"
done
```

Then rerun aggregation/filtering for `phase3_regime_place` rows and merge with `phase2_remote_results/analysis_current/phase2_protocol_aggregate.csv`.
