# Phase-4 Context Pack

Generated: 2026-05-19

## Goal

Run the Phase-4 full matrix while reusing exact-matching Phase-2/3 results. Do not rerun any seed-run that matches the canonical key and protocol.

## Canonical Matrix

- Datasets: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Solar1`-`Solar8`.
- Dataset-horizon cells: `36`.
- Configs: `gelu_all`, `relu_all`, `swish_all`, `tanh_all`, `softsign_all`, `tanh_sin001_all`, `enc_tanh_dec_gelu`, `enc_gelu_dec_tanh`, `tanh_all_outtanh`.
- Seeds: `2024`, `2025`, `2026`.
- Required seed-runs: `972`.
- Protocol: `batch_size=32`, `train_epochs=6`, `embed=timeF`, `attn=prob`, `d_model=512`, `n_heads=8`, `d_ff=2048`, dropout `0.05`.

Canonical matrix file:

- `.aris/phase4_expected_matrix_20260519_phase4_preflight.csv`

## Reuse Rules

Reuse only if all are true:

- Same dataset, horizon, seq_len, label_len, e_layers, d_layers, factor, batch_size, train_epochs, activation_signature, seed.
- `mse` and `mae` are finite.
- `config.json` is readable.
- `batch_size=32`, `train_epochs=6`, `embed=timeF`, `attn=prob`.
- Not marked or named as batch-adjusted/small-batch (`bs4`, `bs8`, `lee_bs`, `batch_adjusted`).

Do not reuse Phase-1 probes, Lee small-batch diagnostics, aggregate-only rows, or mismatched configs.

Reuse outputs:

- `.aris/phase4_reusable_runs_20260519_phase4_preflight.csv`: `116` seed-runs.
- `.aris/phase4_missing_runs_20260519_phase4_preflight.csv`: `856` seed-runs.
- `.aris/phase4_reuse_report_20260519_phase4_preflight.md`.

## Data Manifest

Local data integrity passed 12/12:

- `lee_ocil/ETT-small/ETTh1.csv`
- `lee_ocil/ETT-small/ETTh2.csv`
- `lee_ocil/ETT-small/ETTm1.csv`
- `lee_ocil/ETT-small/ETTm2.csv`
- `Solar/Site_1_50MW.csv`
- `Solar/Site_2_130MW.csv`
- `Solar/Site_3_30MW.csv`
- `Solar/Site_4_130MW.csv`
- `Solar/Site_5_110MW.csv`
- `Solar/Site_6_35MW.csv`
- `Solar/Site_7_30MW.csv`
- `Solar/Site_8_30MW.csv`

Integrity file:

- `.aris/phase4_data_integrity_20260519_phase4_preflight.csv`

## Remote Hosts

Disabled:

- `10.20.12.248`
- `10.20.12.247`

Candidates:

- `10.21.53.62`
- `10.21.53.82`
- `10.21.53.113`
- `10.21.53.142`
- `10.21.53.162`

Current status:

- SSH key-based auth failed on all candidates.
- No transient password env var was present.
- GPU inventory is therefore blocked and recorded as auth-blocked in `.aris/phase4_gpu_inventory_20260519_phase4_preflight.md`.

Resume with:

```bash
PHASE4_SSH_PASSWORD='<transient>' scripts/phase4_remote_control.sh inventory
PHASE4_SSH_PASSWORD='<transient>' scripts/phase4_remote_control.sh sync
PHASE4_SSH_PASSWORD='<transient>' scripts/phase4_remote_control.sh launch .aris/phase4_queue_20260519_phase4_preflight
```

## Queue Manifests

Generated a smoke queue and a post-smoke main queue. Run smoke first, pull results, regenerate the missing matrix, and only then launch the main queue.

Smoke manifests:

- `.aris/phase4_queue_20260519_phase4_preflight_smoke/phase4_queue_manifest_10_21_53_82.json`: 1 job.
- `.aris/phase4_queue_20260519_phase4_preflight_smoke/phase4_queue_manifest_10_21_53_113.json`: 1 job.
- `.aris/phase4_queue_20260519_phase4_preflight_smoke/phase4_queue_manifest_10_21_53_142.json`: 1 job.
- `.aris/phase4_queue_20260519_phase4_preflight_smoke/phase4_queue_manifest_10_21_53_162.json`: 1 job.

Post-smoke main manifests:

- `.aris/phase4_queue_20260519_phase4_preflight/phase4_queue_manifest_10_21_53_62.json`: 216 jobs.
- `.aris/phase4_queue_20260519_phase4_preflight/phase4_queue_manifest_10_21_53_82.json`: 227 jobs.
- `.aris/phase4_queue_20260519_phase4_preflight/phase4_queue_manifest_10_21_53_113.json`: 146 jobs.
- `.aris/phase4_queue_20260519_phase4_preflight/phase4_queue_manifest_10_21_53_142.json`: 108 jobs.
- `.aris/phase4_queue_20260519_phase4_preflight/phase4_queue_manifest_10_21_53_162.json`: 155 jobs.

The smoke queue covers 4 missing seed-runs. The post-smoke main queue excludes those rows and covers the remaining 852 missing seed-runs.

Launch helper fixes:

- Manifests use concrete `server_id` and `server_ip` values, so queue completion checks match the actual result directory names.
- Manifests use `/home/testsv/project/osc_informer/lee_ocil` as remote `cwd` because the queue manager quotes `cwd`.
- Password auth support remains transient through `PHASE4_SSH_PASSWORD`; no password is written to files.

## Autonomous Decisions

- OOM: retry once through the queue. If it still fails, rerun manually at batch size 16 and mark as batch-adjusted; do not mix batch-adjusted runs into the main table.
- Missing row: regenerate matrix and launch only exact missing seed-runs.
- NaN/divergence: rerun once; if repeated, mark invalid in the completion matrix.
- SSH failure: backoff and retry; switch host if still blocked.
- Disk tight: preserve summary/log/config first; arrays only where needed for factor/oracle coverage.

## Partial Results Available Now

Full-grid aggregation over reusable rows only:

- `phase4_remote_results/analysis_current/phase4_protocol_raw_dedup.csv`
- `phase4_remote_results/analysis_current/phase4_protocol_aggregate.csv`
- `phase4_remote_results/analysis_current/phase4_completion_matrix.csv`
- `phase4_remote_results/analysis_current/phase4_paired_vs_gelu.csv`
- `phase4_remote_results/analysis_current/phase4_claim_summary.md`

Partial factor/oracle diagnostics over array-ready reused rows:

- `phase4_remote_results/analysis_current/factors/phase4_factor_sample_results.csv`
- `phase4_remote_results/analysis_current/factors/phase4_factor_bin_summary.csv`
- `phase4_remote_results/analysis_current/factors/phase4_factor_contrast_summary.csv`
- `phase4_remote_results/analysis_current/factors/phase4_static_activation_ranking.csv`
- `phase4_remote_results/analysis_current/factors/phase4_factor_oracle_summary.csv`
- `phase4_remote_results/analysis_current/factors/phase4_factor_oracle_loso_summary.csv`
- `phase4_remote_results/analysis_current/factors/phase4_factor_bin_best_activation.csv`
- `phase4_remote_results/analysis_current/factors/phase4_factor_claim_summary.md`

## Claim Boundary

Current evidence supports only a scoped claim: activation choice is dataset- and horizon-sensitive under matched Informer settings. The broad Phase-4 claim is still pending because `856` required seed-runs are not complete.

## Git Milestone Mapping

Implemented locally:

- `phase4: add autonomous context and full-grid support`
- `phase4: add dataset integrity and factor coverage`
- `phase4: record reusable and missing run matrix`
- `phase4: record gpu inventory and smoke results`
- `phase4: record full-grid launch manifests`
- `phase4: aggregate full-grid results` (partial over reusable rows)
- `phase4: add factor-bin and oracle analyses` (partial over array-ready reused rows)

Blocked until remote auth/launch:

- `phase4: collect remote summaries`
- final full-grid aggregation after all missing runs complete
- `paper: update full dataset evidence`
