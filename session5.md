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

## 2026-05-21 PatchTST Transfer Continuation

PatchTST code was instrumented for activation-transfer experiments under
`PatchTST/PatchTST_supervised/`. The important activation wiring fix was that
`models/PatchTST.py` must read `configs.activation`; otherwise non-GELU configs
silently run with GELU in the backbone.

Cloud execution used four RTX 4090 hosts:

- `10.21.53.82`
- `10.21.53.113`
- `10.21.53.142`
- `10.21.53.162`

Disabled or excluded hosts:

- `10.20.12.248`
- `10.20.12.247`
- `10.21.53.62` due NVML mismatch in earlier checks

Stage1 PatchTST transfer:

- Run id: `patchtst_stage1_20260521_actfix`
- Matrix: 12 dataset-horizon cells x 4 configs x 3 seeds
- Seed-runs: `144/144` completed
- Complete 3-seed config cells: `48/48`
- Result root: `patchtst_remote_results/stage1_actfix_20260521/`
- Analysis root: `patchtst_remote_results/analysis_patchtst_stage1_actfix/`

Stage1 result:

- Non-GELU best cells: `5/12`
- GELU best cells: `7/12`
- `softsign_ffn_linear`: best `3/12`
- `tanh_sin001_ffn_linear`: best `2/12`
- `tanh_ffn_linear`: best `0/12`

Non-GELU wins are limited to ETTh2/720, Solar2/24, Solar2/96, Solar5/24, and
Solar5/96. ETTh2/168, ETTh2/336, all tested ETTm2 horizons, and both Solar3
horizons remain GELU-best.

Stage2 PatchTST placement:

- Run id: `patchtst_stage2_20260521_actfix`
- Matrix: 6 dataset-horizon cells x 3 output-tanh configs x 3 seeds
- Seed-runs: `54/54` completed
- Complete 3-seed config cells: `18/18`
- Result root: `patchtst_remote_results/stage2_actfix_20260521/`
- Analysis root: `patchtst_remote_results/analysis_patchtst_stage2_actfix/`

Stage2 result:

- Output-tanh paired wins: `0/54`
- Mean gain vs matching linear head:
  - GELU: `-5.44%`
  - tanh: `-5.28%`
  - tanh_sin001: `-5.31%`

Combined transfer analysis:

- `patchtst_remote_results/analysis_patchtst_transfer_actfix/patchtst_transfer_claim_summary.md`
- `patchtst_remote_results/analysis_patchtst_transfer_actfix/patchtst_output_tanh_paired.csv`
- `patchtst_remote_results/analysis_patchtst_transfer_actfix/patchtst_transfer_best_cells.csv`
- `.aris/patchtst_reuse_report_20260521_stage1_actfix_complete.md`
- `.aris/patchtst_reuse_report_20260521_stage2_actfix_complete.md`
- `paper/patchtst_transfer_results.md`
- `.aris/patchtst_experiment_audit_20260521.md`
- `.aris/patchtst_result_to_claim_20260521.md`

Claim boundary:

- Supported: PatchTST has activation sensitivity in selected cells.
- Supported: transfer is partial and architecture-dependent.
- Not supported: Informer activation ranking directly transfers to PatchTST.
- Not supported: output tanh is a robust cross-architecture placement rule.

Paper update:

- Updated `paper/story_results_phase4.md`.
- Updated `paper/submission_hypothesis_draft.md`.
- Updated `paper/submission_hypothesis_full.tex`.
- Added `paper/figures/TABLE_patchtst_transfer.tex`.
- Added `paper/figures/TABLE_patchtst_output_tanh.tex`.
- Compiled `paper/submission_hypothesis_full.pdf` with `tectonic
  submission_hypothesis_full.tex`; local `pdflatex`, `latexmk`, and `bibtex`
  are not installed, but `tectonic` completed successfully.

## 2026-05-21 iTransformer Activation Transfer Probe

User requested a quick iTransformer migration test after PatchTST showed weak
transfer. All training was run on cloud RTX 4090 hosts only.

Scope:

- Model repo: `iTransformer/`
- Datasets: `ETTh2`, `ETTm2`
- Horizons: `96`, `336`, `720`
- Configs: `gelu_ffn_linear`, `tanh_ffn_linear`, `softsign_ffn_linear`,
  `tanh_sin001_ffn_linear`
- Seed: `2024`
- Protocol: `seq_len=96`, `label_len=48`, `d_model=128`, `d_ff=128`,
  `batch_size=32`, `train_epochs=10`
- Placement: FFN activation only; output-tanh placement was not tested.

Implementation notes:

- Added iTransformer activation support for `tanh`, `softsign`, `swish/silu`,
  and `tanh_sin001`.
- Added `random_seed`, run metadata, and per-run summary/config/pred/true
  artifact writing to iTransformer.
- Fixed compatibility issues for current NumPy/Pandas: `np.inf`,
  `drop(columns=['date'])`, and minute-frequency alias handling.
- Added iTransformer matrix, manifest, aggregation, sync, and remote-control
  scripts.
- Remote venv lacked `reformer_pytorch`; installed `reformer-pytorch==1.4.4`
  on the four 4090 hosts before relaunching smoke.

Execution:

- Smoke run: `ETTh2/96` GELU and tanh completed with finite metrics.
- Stage1 expected matrix: `24` seed-runs.
- Reused smoke rows: `2`.
- Newly launched stage1 rows: `22`.
- Queue terminal status:
  - `10.21.53.82`: `6/6` completed
  - `10.21.53.113`: `6/6` completed
  - `10.21.53.142`: `5/5` completed
  - `10.21.53.162`: `5/5` completed
- Final deduplicated rows: `24/24` finite.
- Artifact coverage in final result root: `config/metrics/pred/true = 24/24`.

Key results:

- GELU best: `5/6` dataset-horizon cells.
- Non-GELU best: `1/6` cells.
- Only non-GELU best cell: `ETTm2/96`, where
  `tanh_sin001_ffn_linear` MSE `0.183869` vs GELU MSE `0.184226`
  (`+0.194%` paired gain).
- ETTh2/96, ETTh2/336, ETTh2/720, ETTm2/336, and ETTm2/720 are GELU-best.

Claim boundary:

- Not supported: Informer tanh-family activation ranking directly transfers to
  iTransformer.
- Partially supported: activation choice still matters, but the effect is
  architecture- and dataset-horizon-dependent.
- Not tested: output-tanh placement and factor/regime-based placement for
  iTransformer.
- Recommended paper use: negative/weak transfer stress test. Keep strong claims
  scoped to Informer; use PatchTST and iTransformer to support the more
  conservative story that activation behavior is architecture-dependent.

Files:

- Result root: `itransformer_remote_results/stage1_actfix3_20260521/`
- Analysis root: `itransformer_remote_results/analysis_itransformer_stage1_actfix3/`
- Matrix/reuse files: `.aris/itransformer_*_20260521_stage1_actfix3.csv`
- Audit: `.aris/itransformer_experiment_audit_20260521.md`
- Result-to-claim: `.aris/itransformer_result_to_claim_20260521.md`
- Paper note: `paper/itransformer_transfer_results.md`
- Paper table: `paper/figures/TABLE_itransformer_transfer.tex`
- Updated LaTeX draft: `paper/submission_hypothesis_full.tex`
- Updated story/draft notes: `paper/story_results_phase4.md`,
  `paper/submission_hypothesis_draft.md`
- Recompiled `paper/submission_hypothesis_full.pdf` with `tectonic
  submission_hypothesis_full.tex`; compile completed with underfull hbox
  warnings only.

## 2026-05-22 iTransformer Regime/Placement Probe

User requested a small matrix over all currently implemented activation regimes
and placement combinations to test whether default `gelu_ffn_linear` is globally
optimal.

Scope:

- Cells: ETTh2/96, ETTm2/96, ETTm2/720
- FFN regimes: GELU, ReLU, Swish, tanh, softsign, tanh_sin001
- Placements: linear output and output tanh
- Seed: 2024
- Expected rows: `36`
- Reused rows: `12` existing linear rows
- Newly launched rows: `24`
- Remote queue status: four RTX 4090 hosts, `24/24` newly launched jobs
  completed with no Traceback/OOM/NaN alerts
- Aggregated rows: `36/36` finite
- Artifact coverage in `itransformer_remote_results/placeprobe_20260522/`:
  `config/pred/true = 36/36`

Implementation notes:

- Added `--output_activation {linear,tanh}` to iTransformer.
- Applied output tanh after the projector and before iTransformer instance
  de-normalization.
- Updated summary writing to upgrade older `summary.csv` headers before
  appending rows with `output_activation`.
- Updated iTransformer matrix/manifest/aggregation scripts for the
  `placement_probe` stage.

Key result:

- ETTh2/96 best: `gelu_ffn_outtanh`, MSE `0.299667` vs default GELU-linear
  `0.301598`, gain `0.640%`.
- ETTm2/96 best: `tanh_sin001_ffn_linear`, MSE `0.183869` vs default
  `0.184226`, gain `0.194%`.
- ETTm2/720 best: `tanh_ffn_outtanh`, MSE `0.408508` vs default `0.414092`,
  gain `1.348%`.

Interpretation:

- Positive signal found: all three tested cells have a better configuration
  than default `gelu_ffn_linear`.
- Placement is conditional: output tanh helps all six activations on ETTh2/96
  and ETTm2/720, but hurts all six activations on ETTm2/96.
- This supports targeted multi-seed confirmation, not a final paper-level
  iTransformer claim.

Files:

- Result root: `itransformer_remote_results/placeprobe_20260522/`
- Analysis root: `itransformer_remote_results/analysis_itransformer_placeprobe_20260522/`
- Summary: `itransformer_remote_results/analysis_itransformer_placeprobe_20260522/itransformer_placeprobe_summary.md`
- Result-to-claim note: `.aris/itransformer_placeprobe_result_to_claim_20260522.md`
- Paper note updated: `paper/itransformer_transfer_results.md`

## 2026-05-23 Input-Only Factor-Router Feasibility Audit

User clarified that this session should decide whether a data-factor router is
feasible, while model topology/meta-features should be deferred to the next
session.

Implemented an input-only router feasibility audit:

- New script: `lee_ocil/scripts/phase4_router_feasibility.py`
- Regenerated Phase-4 factors with frequency columns:
  `.aris/phase4_timefreq_factors_20260523_router/`
- Confirmed spectral columns are present:
  `spectral_entropy`, `dominant_frequency_energy_ratio`,
  `top3_frequency_energy_ratio`, `low_freq_energy_ratio`,
  `mid_freq_energy_ratio`, `high_freq_energy_ratio`,
  `spectral_centroid`, `spectral_bandwidth`, `spectral_flatness`,
  `daily_cycle_energy_ratio`.
- Added custom input-only router factors including seasonal/harmonic strength,
  dominant-period/horizon ratio, rolling spectral entropy, burstiness, robust
  scale, outlier mass, normalization drift, and cross-channel structure.

Strict leakage boundary:

- Router features are only `input_*` and `router_*`.
- No `target_*`, prediction arrays, errors, best-config labels, or held-out
  metrics are used as features.
- Existing target-bin factor summaries remain diagnostics only, not router
  evidence.

Informer-only result:

- Feature rows: `36` dataset-horizon cells.
- Input/router feature count: `62`.
- Static baseline by mean rank: `tanh_sin001_all`.
- Static mean gain vs GELU: `0.1237`.
- Oracle mean gain vs GELU: `0.1597`.
- Static mean regret vs oracle: `0.0472`.
- LOCO nested single-factor router:
  - mean gain vs GELU: `0.1263`
  - mean gain vs static: `0.00035`
  - positive-vs-static rate: `0.2778`
  - top-1 hit rate: `0.1944`
- LODO nested single-factor router:
  - mean gain vs static: `0.00110`
  - positive-vs-static rate: `0.25`
- Leave-family ETT/Solar router fails:
  - mean gain vs static: `-0.1285`
  - top-1 hit rate: `0.0833`
- Fixed-feature permutation check:
  - observed LOCO gain vs static: `0.0004`
  - permutation mean: `-0.0239`
  - p(perm >= observed): `0.01`

Verdict:

- NO-GO for a deployable Informer-only zero-shot router.
- Important nuance: this does not mean factors are useless. It means
  `tanh_sin001_all` is already a very strong Informer static policy, so the
  fair router target should not be "beat Informer tanh_sin001 on Informer."

Cross-architecture context:

- New script: `lee_ocil/scripts/phase4_cross_arch_router_context.py`
- Output root: `phase4_remote_results/analysis_current/factors_router/`
- Summary: `phase4_cross_arch_router_context.md`

Cross-architecture result:

| model | cells | non-baseline best cells | best static | mean best gain vs baseline |
| --- | ---: | ---: | --- | ---: |
| Informer | 36 | 34 | `tanh_sin001_all` | 0.1597 |
| PatchTST | 12 | 5 | `gelu_ffn_linear` | 0.0038 |
| iTransformer-stage1 | 6 | 1 | `gelu_ffn_linear` | 0.0003 |
| iTransformer-placeprobe | 3 | 3 | `swish_ffn_outtanh` by mean rank | 0.0073 |

Interpretation:

- User's intuition is confirmed: in Informer, `tanh_sin001_all` is so strong
  that router gains over it are naturally small.
- PatchTST is a negative transfer stress test: GELU-linear is best in most
  cells and output tanh is harmful in all paired placement tests.
- iTransformer FFN-only stage mostly keeps GELU best, but the placement probe
  finds positive signals in all three tested cells, especially output-tanh on
  ETTh2/96 and ETTm2/720.
- The next session should therefore study a model-aware router:
  data factors + architecture/topology metadata should decide whether
  tanh-family transfers, whether GELU should remain, and whether output
  placement should be enabled.

Files:

- `phase4_remote_results/analysis_current/factors_router/phase4_router_factor_table.csv`
- `phase4_remote_results/analysis_current/factors_router/phase4_router_correlation.csv`
- `phase4_remote_results/analysis_current/factors_router/phase4_router_eval_summary.csv`
- `phase4_remote_results/analysis_current/factors_router/phase4_router_feasibility_summary.md`
- `phase4_remote_results/analysis_current/factors_router/phase4_cross_arch_model_summary.csv`
- `phase4_remote_results/analysis_current/factors_router/phase4_cross_arch_static_ranking.csv`
- `phase4_remote_results/analysis_current/factors_router/phase4_cross_arch_best_cells.csv`
- `phase4_remote_results/analysis_current/factors_router/phase4_cross_arch_router_context.md`

## 2026-05-23 Model-Aware Router Attempt

User noted that Informer-only routing is hard because `tanh_sin001_all` is
nearly the strongest static Informer policy, and requested that PatchTST and
iTransformer results be considered together.

Implemented:

- New script: `lee_ocil/scripts/phase4_model_aware_router.py`
- Feature ablation script:
  `lee_ocil/scripts/phase4_model_aware_router_ablation.py`
- Main deployable output root:
  `phase4_remote_results/analysis_current/factors_router/model_aware/`
- Action-class oracle upper-bound root:
  `phase4_remote_results/analysis_current/factors_router/model_aware_action_oracle/`
- Claim summary:
  `phase4_remote_results/analysis_current/factors_router/phase4_factor_router_claim_summary.md`

Important correction made during the audit:

- The first model-aware run evaluated the best config inside the predicted
  action class on the held-out cell. That is an action-class upper bound, not a
  deployable zero-shot router.
- The final main run uses a deployable config-level protocol: the training fold
  selects a concrete config, and the held-out cell is evaluated only with that
  config.

Model-aware action-class upper bound:

- Cells: `57`
- Leave-cell mean gain vs static: `0.0198`
- Leave-cell positive-vs-static rate: `0.4912`
- Leave-cell mean regret vs oracle: `0.00775`
- Leave-dataset mean gain vs static: `0.0167`
- Verdict: `PARTIAL-GO` only as an action-class upper bound.

Deployable exact-config router:

- Cells: `57`
- Leave-cell mean gain vs static: `0.00233`
- Leave-cell positive-vs-static rate: `0.2982`
- Leave-cell config hit rate: `0.3684`
- Leave-dataset mean gain vs static: `0.00225`
- Leave-dataset positive-vs-static rate: `0.2807`
- Leave-dataset config hit rate: `0.2982`
- Leave-model-scope and leave-family fail because unseen model scopes fall
  back to baseline.
- Verdict: `NO-GO` for a deployable exact-config zero-shot router.

Feature-set ablation under deployable config protocol:

| feature set | leave-cell gain vs static | leave-dataset gain vs static |
| --- | ---: | ---: |
| constant scope rule | -0.00086 | 0.00308 |
| data only | -0.00060 | 0.00258 |
| frequency only | -0.00128 | 0.00187 |
| statistical only | -0.00060 | 0.00270 |
| topology only | 0.00233 | 0.00308 |
| model ID only | -0.00086 | 0.00308 |
| all features | 0.00233 | 0.00308 |

Interpretation:

- User's Informer intuition is correct in the static-policy sense:
  `tanh_sin001_all` is the best mean-rank static Informer policy, although it
  is not the best cell-wise config in every cell.
- PatchTST and iTransformer change the story: activation transfer is
  architecture-dependent, not globally tanh-family.
- Data/time-frequency factors have diagnostic signal, but current deployable
  router performance is mostly explained by architecture/topology or simple
  scope-level behavior.
- Current paper should not claim a final zero-shot exact activation/placement
  router.
- Defensible story: architecture/topology gate first, then data-factor
  refinement within each architecture; this needs a balanced cross-architecture
  follow-up matrix.

Next minimal experiments:

- Expand iTransformer placement probes to 3 seeds and more cells.
- Confirm PatchTST negative transfer with 3 seeds for bounded-linear vs
  GELU-linear, avoiding broad output-tanh unless specifically needed.
- Re-run router after balancing model-cell counts, because current
  cross-architecture pool is dominated by 36 Informer cells.

Follow-up attempt:

- Added `lee_ocil/scripts/phase4_scope_local_router.py` to test a two-stage
  router: model-scope gate first, then data-factor refinement inside each seen
  scope.
- Output roots:
  - `phase4_remote_results/analysis_current/factors_router/scope_local_data_abstain/`
  - `phase4_remote_results/analysis_current/factors_router/scope_local_action_abstain/`

Scope-local exact-config refinement results:

| feature set | leave-cell gain vs train-scope static | leave-dataset gain vs train-scope static |
| --- | ---: | ---: |
| data/statistical factors | `0.0135` | `0.0020` |
| frequency-only factors | `-0.0063` | `-0.0169` |

Scope-local action-level refinement results:

| feature set | leave-cell gain vs train-scope static | leave-dataset gain vs train-scope static |
| --- | ---: | ---: |
| frequency-only action rule | `0.0006` | `-0.0005` |
| data/statistical action rule | `-0.0003` | `-0.0042` |

Interpretation update:

- Two-stage seen-scope refinement has a small positive signal for exact-config
  data/statistical factors, especially within Informer.
- Frequency-only factors are not sufficient in the current implementation.
- Action-level routing is weaker than exact-config refinement, so the current
  signal is not simply bounded-vs-GELU-vs-output-tanh.
- This still does not support a final deployable cross-architecture router, but
  it supports keeping the two-stage router hypothesis alive for the next,
  balanced cross-architecture experiment set.

## 2026-05-23 Pairwise Regime/Placement Factor Router

User requested another attempt focused specifically on using factors to judge
placement and regime, after checking how related work treats this type of
problem.

Method shift:

- Reframed activation routing as pairwise uplift estimation, matching the
  algorithm-selection / time-series meta-learning pattern.
- Instead of predicting the globally best config, each rule asks whether a
  specific action should be enabled:
  `uplift = (MSE(comparator) - MSE(action)) / MSE(comparator)`.
- A rule recommends an action only when a single factor threshold fires;
  otherwise it abstains.

Implemented:

- `lee_ocil/scripts/phase4_pairwise_factor_router.py`
- Output root:
  `phase4_remote_results/analysis_current/factors_router/pairwise_uplift/`
- Method report:
  `phase4_remote_results/analysis_current/factors_router/phase4_pairwise_factor_router_method_report.md`

Pair rows:

- Total pairwise action rows: `735`.
- Regime pairs: bounded/tanh/tanh_sin/softsign/swish/relu vs default or
  within-family comparators.
- Placement pairs: output tanh, encoder tanh, decoder tanh, all-tanh vs split.

Key pair availability:

| action | pairs | mean uplift | positive rate |
| --- | ---: | ---: | ---: |
| `regime_best_bounded_vs_default` | 57 | `0.0931` | `0.6491` |
| `regime_tanh_vs_default` | 57 | `0.0779` | `0.6316` |
| `regime_tanh_sin_vs_default` | 57 | `0.0767` | `0.6140` |
| `placement_encoder_tanh_vs_default` | 36 | `0.0573` | `0.6944` |
| `placement_all_tanh_vs_best_split` | 36 | `0.0554` | `0.6111` |
| `placement_decoder_tanh_vs_default` | 36 | `0.0242` | `0.6111` |
| `placement_output_tanh_vs_best_linear` | 45 | `-0.4753` | `0.1556` |

Most important result:

- `placement_all_tanh_vs_best_split` is the clearest factor-conditioned
  placement signal.
- Leave-dataset, frequency features:
  - policy uplift: `0.0668`
  - always-action uplift: `0.0554`
  - excess over always-action: `+0.0114`
  - recommend rate: `0.5556`
  - precision if recommended: `0.8500`
  - mean uplift if recommended: `0.1202`
  - selected factors:
    `router_timefreq_rolling_spectral_entropy_std`,
    `router_timefreq_burstiness_index`
- Interpretable rule:
  low rolling spectral-entropy variability favors full/all-tanh placement over
  split encoder/decoder placement.

Other placement findings:

- `placement_encoder_tanh_vs_default` is broadly useful:
  mean uplift `0.0573`, positive rate `0.6944`.
  Frequency factors provide high-confidence filtering but do not clearly beat
  always enabling encoder tanh.
- `placement_decoder_tanh_vs_default` is weaker:
  mean uplift `0.0242`, positive rate `0.6111`.
- Output tanh is globally harmful:
  matched output-tanh mean uplift `-0.4248`;
  output-tanh vs best-linear mean uplift `-0.4753`.
  Frequency rules identify a tiny safe subset, but this is not enough for a
  paper-level output-tanh router claim.

Regime findings:

- Bounded regimes are strong as a static/architecture-conditioned policy.
- Factor rules can identify many positive bounded-regime cases, using factors
  such as `input_volatility`, `input_turbulence_score`,
  `router_freq_seasonal_peak_strength`, and `input_daily_cycle_energy_ratio`.
- However, because bounded regimes are already often good, data-factor rules do
  not reliably beat always trying bounded regimes.
- Topology/model features still explain regime selection better than pure data
  factors in the combined cross-architecture matrix.

Updated claim boundary:

- Safe: factors can judge some placement decisions, especially all-tanh vs
  split placement under stable time-frequency structure.
- Safe: factors provide confidence for bounded-regime use.
- Unsafe: current factors can generally select the best activation function and
  placement zero-shot across architectures.
