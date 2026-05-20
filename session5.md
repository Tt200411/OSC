# Session 5 - Phase-4 Full-Grid Setup

Date: 2026-05-19

## Objective

Extend Phase-2/3 results toward the full Phase-4 matrix without rerunning seed-runs that already match the canonical protocol. Main-table eligibility remains strict: a dataset-horizon-config cell needs seeds `2024`, `2025`, and `2026` under the same protocol.

## Implemented

- Added a Phase-4 runner that supports `ETTm1`, `ETTm2`, and `Solar1` through `Solar8`: `lee_ocil/phase4_informer.py`.
- Added canonical Phase-4 matrix/reuse tooling:
  - `lee_ocil/scripts/phase4_common.py`
  - `lee_ocil/scripts/phase4_matrix.py`
  - `lee_ocil/scripts/build_phase4_queue_manifest.py`
  - `lee_ocil/scripts/phase4_aggregate.py`
  - `lee_ocil/scripts/phase4_factor_pipeline.py`
  - `lee_ocil/scripts/generate_phase4_factors.py`
- Added remote helpers:
  - `scripts/sync_phase4_remote.sh`
  - `scripts/phase4_remote_control.sh`
- Patched `lee_ocil/scripts/analyze_regime_activation_oracle.py` for pandas categorical-bin edge cases.

## Preflight Results

- Local data integrity: 12/12 CSV files present with expected schema and target column.
- Expected matrix: `972` required seed-runs across `36` dataset-horizon cells.
- Strictly reusable Phase-2/3 seed-runs: `116`.
- Missing seed-runs to launch: `856`.
- Complete three-seed cells available before Phase-4 launches: `37` dataset-horizon-config cells.

Key files:

- `.aris/phase4_expected_matrix_20260519_phase4_preflight.csv`
- `.aris/phase4_reusable_runs_20260519_phase4_preflight.csv`
- `.aris/phase4_missing_runs_20260519_phase4_preflight.csv`
- `.aris/phase4_reuse_report_20260519_phase4_preflight.md`
- `.aris/phase4_latest.json`

## Queue Manifests

The launch plan is split into a smoke queue and a post-smoke main queue.

Smoke manifests were generated in `.aris/phase4_queue_20260519_phase4_preflight_smoke/`.

| host | jobs |
| --- | ---: |
| 10.21.53.82 | 1 |
| 10.21.53.113 | 1 |
| 10.21.53.142 | 1 |
| 10.21.53.162 | 1 |

Post-smoke missing-run manifests were generated in `.aris/phase4_queue_20260519_phase4_preflight/`.

| host | jobs |
| --- | ---: |
| 10.21.53.62 | 216 |
| 10.21.53.82 | 227 |
| 10.21.53.113 | 146 |
| 10.21.53.142 | 108 |
| 10.21.53.162 | 155 |

The smoke queue covers four missing smoke runs. The main queue excludes those smoke rows and covers the remaining `852` missing seed-runs. Regenerate the missing matrix after smoke results are collected before launching the main queue if any smoke run fails or becomes invalid.

Launch is blocked in the current shell because all candidate hosts reject key-based SSH and no transient password is available. No password was written to disk.

Resume commands once a transient password is available:

```bash
PHASE4_SSH_PASSWORD='<transient>' scripts/phase4_remote_control.sh inventory
PHASE4_SSH_PASSWORD='<transient>' scripts/phase4_remote_control.sh sync
PHASE4_SSH_PASSWORD='<transient>' scripts/phase4_remote_control.sh launch .aris/phase4_queue_20260519_phase4_preflight_smoke
# After smoke completion and result pull/reuse scan:
PHASE4_SSH_PASSWORD='<transient>' scripts/phase4_remote_control.sh launch .aris/phase4_queue_20260519_phase4_preflight
```

Launch helper fixes made in this session:

- Queue manifests now use concrete `server_id`/`server_ip` values, so expected completion paths match generated result directories.
- Queue manifests now use `/home/testsv/project/osc_informer/lee_ocil` instead of a shell `$HOME` expression, because the queue manager shell-quotes `cwd`.
- Sync and inventory helpers support transient password auth without writing credentials to disk.

## Partial Analysis Before New Launches

Current reusable-run aggregation:

- `phase4_remote_results/analysis_current/phase4_protocol_raw_dedup.csv`
- `phase4_remote_results/analysis_current/phase4_protocol_aggregate.csv`
- `phase4_remote_results/analysis_current/phase4_completion_matrix.csv`
- `phase4_remote_results/analysis_current/phase4_paired_vs_gelu.csv`
- `phase4_remote_results/analysis_current/phase4_claim_summary.md`

Current array-ready factor/oracle analysis:

- `phase4_remote_results/analysis_current/factors/phase4_factor_sample_results.csv` (`563712` rows)
- `phase4_remote_results/analysis_current/factors/phase4_factor_bin_summary.csv` (`493` rows)
- `phase4_remote_results/analysis_current/factors/phase4_factor_contrast_summary.csv` (`174` rows)
- `phase4_remote_results/analysis_current/factors/phase4_static_activation_ranking.csv` (`20` rows)
- `phase4_remote_results/analysis_current/factors/phase4_factor_oracle_summary.csv` (`84` rows)
- `phase4_remote_results/analysis_current/factors/phase4_factor_oracle_loso_summary.csv` (`84` rows)
- `phase4_remote_results/analysis_current/factors/phase4_factor_bin_best_activation.csv` (`252` rows)
- `phase4_remote_results/analysis_current/factors/phase4_factor_claim_summary.md`

These factor/oracle outputs are partial diagnostics only. They cover reused rows with arrays, not the full Phase-4 matrix.

## Current Claim Boundary

Supported by complete reused cells only: activation choice is dataset- and horizon-sensitive under matched Informer settings. Not supported yet: any universal bounded/oscillatory activation claim over all datasets. Small-batch Lee rows remain excluded from the main table.

## Blocker

Remote execution cannot proceed in this context because SSH needs password authentication. Use only transient environment or interactive expect when resuming; do not put passwords in files, scripts, logs, git commits, or command history.

## 2026-05-19 Venv Resume Update

The SSH password was provided transiently by the user and was only passed through interactive stdin/environment for each command. No password is stored in scripts, logs, notes, or manifests.

Remote runtime checks found that the candidate hosts do not have a usable conda installation for this project, but they do have a project venv at `/home/testsv/project/osc_informer/lee_ocil/.venv`. Phase-4 queue launch was switched to venv mode.

Venv smoke completed on all four smoke jobs:

| host | smoke job | status |
| --- | --- | --- |
| 10.21.53.82 | Solar2/96 gelu seed2024 | completed |
| 10.21.53.113 | Solar8/96 gelu seed2024 | completed |
| 10.21.53.142 | ETTm1/96 gelu seed2024 | completed |
| 10.21.53.162 | ETTm2/96 gelu seed2024 | completed |

Smoke artifacts were pulled to `phase4_remote_results/smoke_venv_20260519/`. The post-smoke reuse scan is:

- Expected seed-runs: `972`
- Reusable seed-runs: `120`
- Missing seed-runs: `852`
- Data integrity: `12/12` local CSVs valid

Current post-smoke matrix files:

- `.aris/phase4_expected_matrix_20260519_phase4_post_smoke_venv2.csv`
- `.aris/phase4_reusable_runs_20260519_phase4_post_smoke_venv2.csv`
- `.aris/phase4_missing_runs_20260519_phase4_post_smoke_venv2.csv`
- `.aris/phase4_reuse_report_20260519_phase4_post_smoke_venv2.md`

Main full-grid manifests were rebuilt from the post-smoke missing matrix in `.aris/phase4_queue_20260519_phase4_full_venv_post_smoke/`:

| host | jobs |
| --- | ---: |
| 10.21.53.62 | 216 |
| 10.21.53.82 | 227 |
| 10.21.53.113 | 146 |
| 10.21.53.142 | 108 |
| 10.21.53.162 | 155 |

Full-grid launch notes:

- Active queue list: `.aris/phase4_active_queues_20260519.txt`
- `10.21.53.62` first full-grid queue used a venv `activate` file path that was unavailable at job runtime, causing non-training failures. The queue manager was patched to use `VIRTUAL_ENV`/`PATH` instead of sourcing `activate`, and `.62` was relaunched as `20260519_phase4_full_venv62_retry`.
- The old `.62` queue should be ignored for aggregation; use the active queue list above.
- Other four hosts continue under `20260519_phase4_full_venv`.

Latest monitored state showed all five active queues progressing with no OOM/Traceback/NaN alerts. The `.62` retry queue is healthy.

## 2026-05-21 Full-Grid Collection And Solar4 Nanfix

Full-grid queue results were pulled to `phase4_remote_results/full_venv_20260521/`.
The strict reuse scan after collection found:

- Expected seed-runs: `972`
- Reusable seed-runs: `918`
- Missing seed-runs: `54`

All missing canonical seed-runs were `Solar4` for horizons `24` and `96`
across the 9 configs and 3 seeds. The remote `Solar4` queue rows completed,
but had non-finite metrics; job logs showed NaN losses from epoch 1.

Root cause was a data issue in `Solar/Site_4_130MW.csv`: six missing values
in the target column `Power`. The six values were filled in place using a
minimal local policy: nighttime zero-irradiance gaps were filled with `0.0`,
and daytime gaps were filled by immediate-neighbor linear interpolation. The
repair audit is recorded in `.aris/phase4_solar4_nanfix_20260521.md`.

After the repair:

- `Solar4` total NaN: `0`
- `Solar4 Power` NaN: `0`
- Solar4 SHA256: `47c851e6e04642770cd94e76e5b6782bcf4203fb08cf4695a8d08fa6bbd70672`

Next action: sync the repaired dataset to candidate hosts, relaunch only the
54 missing Solar4 seed-runs, fetch summaries, and rerun the strict reuse scan.

## 2026-05-21 Completion, Analysis, And Paper Update

The repaired Solar4 dataset was synced to the candidate hosts and only the 54
missing canonical Solar4 seed-runs were relaunched. The relaunch completed with
no stuck jobs and no NaN alerts. Results were pulled to
`phase4_remote_results/solar4_nanfix_20260521/`.

The final strict reuse scan is:

- Expected seed-runs: `972`
- Reusable seed-runs: `972`
- Missing seed-runs: `0`
- Data integrity: `12/12` local CSVs valid, `0` target NaNs, `0` total NaNs
- Complete dataset-horizon-config cells: `324/324`

Current final matrix files:

- `.aris/phase4_expected_matrix_20260521_phase4_full_complete_arrays_pref.csv`
- `.aris/phase4_reusable_runs_20260521_phase4_full_complete_arrays_pref.csv`
- `.aris/phase4_missing_runs_20260521_phase4_full_complete_arrays_pref.csv`
- `.aris/phase4_reuse_report_20260521_phase4_full_complete_arrays_pref.md`
- `.aris/phase4_data_integrity_20260521_phase4_full_complete_arrays_pref.csv`

Main aggregation files:

- `phase4_remote_results/analysis_current/phase4_protocol_raw_dedup.csv`
- `phase4_remote_results/analysis_current/phase4_protocol_aggregate.csv`
- `phase4_remote_results/analysis_current/phase4_completion_matrix.csv`
- `phase4_remote_results/analysis_current/phase4_paired_vs_gelu.csv`
- `phase4_remote_results/analysis_current/phase4_claim_summary.md`

Full-matrix factor-bin files:

- `phase4_remote_results/analysis_current/factors/phase4_factor_cell_summary.csv`
- `phase4_remote_results/analysis_current/factors/phase4_factor_bin_summary.csv`
- `phase4_remote_results/analysis_current/factors/phase4_factor_contrast_summary.csv`
- `phase4_remote_results/analysis_current/factors/phase4_static_activation_ranking.csv`
- `phase4_remote_results/analysis_current/factors/phase4_factor_bin_best_activation.csv`
- `phase4_remote_results/analysis_current/factors/phase4_factor_claim_summary.md`

Sample-level oracle files remain array-limited diagnostics:

- `phase4_remote_results/analysis_current/factors/phase4_factor_sample_results.csv`
- `phase4_remote_results/analysis_current/factors/phase4_factor_oracle_summary.csv`
- `phase4_remote_results/analysis_current/factors/phase4_factor_oracle_loso_summary.csv`

Result-to-claim boundary:

- Supported: activation choice is dataset- and horizon-sensitive under matched
  Phase-4 Informer settings.
- Supported with scope: bounded and tanh-family activations are strong static
  choices by mean rank.
- Not supported: a universal activation, a universal bounded/oscillatory win,
  or a deployable factor-conditioned activation router.
- Dataset-specific: Solar3 remains GELU-best at both horizons.

Paper update:

- Added Phase-4 figure/table generation in `figures/gen_phase4_paper_assets.py`.
- Generated Phase-4 coverage, best-activation, static-ranking, and LOSO-oracle
  tables plus full-grid heatmap/ranking figures.
- Updated `paper/main.tex` and sections to use the full-grid evidence.
- Compiled `paper/main.pdf` successfully with `tectonic --keep-logs main.tex`
  because `latexmk`/`pdflatex` are not installed locally.
