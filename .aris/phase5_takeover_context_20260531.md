# Phase-5 Takeover Context

Date: 2026-05-31
Workspace: `/Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main`

## Purpose

This file freezes the current project state before any new Phase-5 experiment or paper rewrite. The takeover policy is additive only: do not delete, reset, clean, or rewrite existing `.aris/`, paper, result, model, or remote-control artifacts unless the user explicitly asks for that exact destructive action.

## Local Verification

Checked `.aris/phase4_latest.json` on 2026-05-31:

- `required_seed_runs`: 972
- `reusable_seed_runs`: 972
- `missing_seed_runs`: 0
- `data_ok`: true
- current strict reuse report: `.aris/phase4_reuse_report_20260521_phase4_full_complete_arrays_pref.md`

The git worktree is intentionally dirty and contains many modified/untracked research artifacts. Treat them as valuable project state.

## Evidence Boundary

### Strong Current Evidence

- Informer Phase-4 is the main evidence base.
- Informer full grid is complete: 36 dataset-horizon cells, 9 configs, 3 seeds, 972 seed-runs.
- All 324 dataset-horizon-config cells have complete three-seed coverage.
- Non-GELU activations are best in 34/36 Informer cells.
- All 20 ETT/ETTm cells prefer a non-GELU activation.
- Solar is mixed but mostly non-GELU-best: 14/16 cells.
- Best average static Informer policies:
  - `tanh_sin001_all`, mean rank 2.78
  - `tanh_all`, mean rank 3.19
  - `softsign_all`, mean rank 3.47

### Conservative Paper Claim

Use this as the default claim boundary:

> Under a matched Informer official-script protocol with complete three-seed coverage, activation choice is a dataset- and horizon-sensitive design variable. Bounded and tanh-family activations are strong static choices on average, but the full grid rejects a universal activation or deployable factor-router claim.

### Claims Not Supported

Do not write or imply the following without new multi-seed evidence:

- a universal activation winner;
- universal bounded/oscillatory superiority;
- a deployable exact-config zero-shot activation router;
- direct transfer of the Informer activation ranking to PatchTST or iTransformer;
- output tanh as a robust architecture-independent placement rule;
- reproduction or surpassing of original Informer Table 2.

## Transfer Evidence

### PatchTST

PatchTST is a partial and architecture-dependent transfer result:

- Stage1: non-GELU best in 5/12 cells.
- GELU remains best in 7/12 cells.
- Output-tanh placement is negative in the tested paired matrix: 0/54 paired seed-run wins, about 5.3%-5.4% mean degradation.
- Use PatchTST as a stress test that prevents overclaiming direct transfer.

### iTransformer

iTransformer currently has two evidence levels:

- FFN-only stage1: GELU best in 5/6 tested cells; only ETTm2/96 has a small non-GELU win.
- Placement probe: single-seed positive signal in 3/3 selected cells:
  - ETTh2/96: `gelu_ffn_outtanh`, +0.640% vs default GELU-linear.
  - ETTm2/96: `tanh_sin001_ffn_linear`, +0.194% vs default GELU-linear.
  - ETTm2/720: `tanh_ffn_outtanh`, +1.348% vs default GELU-linear.

This supports targeted three-seed confirmation only. It is not yet a paper-level iTransformer claim.

## Factor-Router Boundary

Current router sample size is 57 model-cells, not 972 seed-runs. The router evidence is diagnostic:

- Informer-only input factor router does not robustly beat the static `tanh_sin001_all` baseline.
- Cross-architecture action-class upper bound is useful diagnostically but is not deployable.
- Cross-architecture exact-config router is no-go as a deployable zero-shot router.
- Data/time-frequency factors have weak-to-moderate diagnostic signal, but architecture/topology is currently stronger.
- A defensible future hypothesis is two-stage routing: architecture/topology gate first, then data-factor refinement within a seen architecture.

## Remote Execution Rules

No deep learning training may run locally except tiny CPU/schema checks.

Allowed cloud hosts for future preflight only:

- `10.21.53.62`
- `10.21.53.82`
- `10.21.53.113`
- `10.21.53.142`
- `10.21.53.162`

Disabled hosts:

- `10.20.12.248`
- `10.20.12.247`

Do not use `10.21.53.156` unless the user explicitly re-enables it and a fresh preflight passes.

SSH credentials must remain transient. Do not write passwords into source files, generated docs, logs, queue manifests, shell scripts, commits, or command notes.

Before any future launch, rerun remote preflight and confirm:

- `nvidia-smi` works;
- no conflicting GPU jobs are present;
- disk has enough free space;
- `/home/testsv/project/osc_informer/lee_ocil/.venv` exists;
- venv imports `torch`, `numpy`, `pandas`, and required analysis/training packages;
- existing queue managers are not active training jobs or are intentionally handled.

## 2026-05-31 Remote Preflight Snapshot

Inventory output directory:

- `.aris/phase5_remote_inventory_20260531/`

Summary:

- `10.21.53.62`: reachable and data complete, but `nvidia-smi` fails with NVML driver/library mismatch. Do not schedule GPU work here until fixed.
- `10.21.53.82`: RTX 4090 visible, 40 MiB used, data complete, venv imports OK. Old Phase-4 smoke queue manager processes are still present and must be treated as stale-process risk before launching.
- `10.21.53.113`: RTX 4090 visible, 537 MiB used, data complete, venv imports OK. One non-project GPU compute app uses about 386 MiB; recheck before launch.
- `10.21.53.142`: RTX 4090 visible, 40 MiB used, data complete, venv imports OK. Several long-running CPU Python processes by another user are present; GPU appears free.
- `10.21.53.162`: RTX 4090 visible, 236 MiB used, data complete, venv imports OK. No project GPU app visible.

Do not launch new work from this snapshot alone if the launch happens later; rerun preflight immediately before launch.

## Next Implementation Entry Points

1. Use `.aris/phase5_experiment_queue_plan_20260531.md` for the next targeted experiment matrix.
2. If writing, update paper claim language only after checking each statement against the evidence boundary above.
3. If new experiments finish, run result-to-claim before strengthening any paper claim.
