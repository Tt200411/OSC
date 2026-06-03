# Phase-5 Router-Core Validation Plan

Date: 2026-05-31

## Main Thesis

Phase-5 的主线是验证 `factor-router` 是否能成为论文核心算法：仅使用输入序列的统计/时频因子，为 Transformer-style Informer 选择 activation regime / placement，在不改网络主体结构的情况下提升预测效果。

该主线不是直接预测 best config，而是：

1. 对每个 action 学习 pairwise uplift rule；
2. 每个 rule 判断“是否启用该 action”；
3. 多个 action 同时触发时，用训练 fold 的 validation uplift / precision 排序；
4. 低置信度或无 action 触发时 abstain 到 `gelu_all` 或 train-fold strongest static。

## Claim Map

| Claim | Evidence Required | Current Status | Paper Use |
| --- | --- | --- | --- |
| C1: Router beats GELU under matched Informer protocol | leave-cell / leave-dataset / leave-horizon gain vs `gelu_all` > 0, complete 3-seed cells | Supported retrospectively: about 12.3% mean gain vs GELU | Main result candidate |
| C2: Router is competitive with strong static baseline | gain vs train-fold strongest static is non-negative or only negligible negative with clear regret explanation | Partial: near tie, slightly negative under leave-dataset / leave-horizon | Claim cautiously; not as “wins static” yet |
| C3: Router captures oracle structure | oracle capture ratio and regret vs oracle are reported | Supported retrospectively: about 62%-64% capture, about 4.7% oracle regret | Mechanism support |
| C4: Router generalizes to new datasets | zero-shot evaluation on Weather / Exchange / ILI without target-test leakage | Not run yet | Required before broad claim |
| C5: Oracle-supplement improves failed zero-shot | small oracle matrix used only for calibration/training, evaluated on held-out split | Not run yet | Backup route if zero-shot fails |

## Fixed Action Space

Regime actions:

- `tanh_all`
- `softsign_all`
- `tanh_sin001_all`

Placement actions:

- `enc_tanh_dec_gelu`
- `enc_gelu_dec_tanh`
- `tanh_all`
- `tanh_all_outtanh`

Default:

- `gelu_all`

Current implementation maps these to pairwise action names in `lee_ocil/scripts/phase4_pairwise_factor_router.py` and can restrict cell-level tables with `--core_only`.

## Retrospective Informer Validation

Input evidence:

- Informer full grid: 36 dataset-horizon cells, 9 configs, 3 seeds.
- `.aris/phase4_latest.json`: `972/972`, `missing_seed_runs=0`, `data_ok=true`.

Run command:

```bash
python lee_ocil/scripts/phase4_pairwise_factor_router.py \
  --factors_dir .aris/phase4_timefreq_factors_20260523_router \
  --output_dir .aris/phase5_router_core_20260531 \
  --scope_filter Informer-fullgrid \
  --core_only \
  --modes leave_cell leave_dataset leave_horizon
```

Required reporting tables:

- Table 2: router vs `gelu_all`, train-fold strongest static, and oracle.
- Table 3: action precision, placement precision, action/config hit rate.
- Table 4: split stability across leave-cell, leave-dataset, leave-horizon.

Generated candidate table file:

- `.aris/phase5_router_core_20260531/phase5_router_cell_tables.md`

## Cross-Dataset Validation

First-wave datasets:

| dataset | horizons | root/data path | target | freq |
| --- | --- | --- | --- | --- |
| Weather | 96, 192, 336 | `../Weather/weather.csv` | `OT` | `t` |
| Exchange | 96, 192, 336 | `../Exchange/exchange_rate.csv` | `OT` | `d` |
| ILI | 24, 36, 48 | `../ILI/national_illness.csv` | `OT` | `w` |

Local status:

- Current local repo does not contain these CSV files.
- Preflight report: `.aris/phase5_new_dataset_preflight_20260531/phase5_new_dataset_preflight.md`.

Smoke rule:

- Local: schema and loader shape only, no training.
- Cloud: dataloader smoke only before queue launch.

Preflight command:

```bash
python lee_ocil/scripts/phase5_dataset_preflight.py \
  --output_dir .aris/phase5_new_dataset_preflight_20260531 \
  --loader_smoke
```

## Oracle Matrix

1-seed oracle matrix:

- datasets: Weather, Exchange, ILI
- horizons: as above
- configs: `gelu_all`, `tanh_sin001_all`, `tanh_all`, `softsign_all`, `enc_tanh_dec_gelu`, `tanh_all_outtanh`
- seed: `2024`
- total: 54 runs

Generated matrix:

- `.aris/phase5_new_dataset_oracle_matrix_20260531.csv`

Generation command:

```bash
python lee_ocil/scripts/phase5_build_router_oracle_matrix.py \
  --output .aris/phase5_new_dataset_oracle_matrix_20260531.csv
```

Follow-up seeds:

- Add `2025,2026` only after 1-seed oracle and zero-shot trend are analyzed.
- The 1-seed results may guide triage, but cannot be written as statistical paper conclusions.

## Decision Gate

GO:

- zero-shot router has stable positive gain vs `gelu_all`;
- vs strongest static is non-negative or negligibly negative with clearly lower oracle regret;
- oracle regret is interpretable by factors/horizon/dataset;
- at least part of Weather / Exchange / ILI supports the same mechanism.

CALIBRATE:

- zero-shot beats `gelu_all` but does not beat static;
- oracle-supplement calibration materially improves held-out performance;
- paper becomes “oracle-supervised / calibrated low-cost activation router”.

PIVOT:

- router only beats GELU and remains far from static/oracle;
- paper language changes to “factor-guided activation search” rather than zero-shot selector.

## Remote Execution Rules

- No deep learning training locally except tiny CPU/schema checks.
- Fresh remote preflight before every launch:
  - `nvidia-smi`
  - disk
  - screen/process queues
  - active GPU apps
  - venv path
  - torch/cuda import check
- Allowed hosts: `10.21.53.82`, `10.21.53.113`, `10.21.53.142`, `10.21.53.162`, and `10.21.53.62` only after NVML mismatch is fixed.
- Disabled hosts: `10.20.12.248`, `10.20.12.247`.
- SSH credentials remain transient only; never write them into files, logs, scripts, manifests, or commits.

## Paper Boundary

Safe first-version wording:

> We propose a zero-shot activation router framework and validate it retrospectively on a complete matched Informer grid. The router learns pairwise uplift rules from input statistical and time-frequency factors, selects activation regime/placement actions, and abstains to conservative defaults when confidence is low.

Required experiment disclosures:

- router vs `gelu_all`;
- router vs train-fold strongest static;
- router regret vs oracle;
- oracle capture ratio;
- abstain rate;
- split definitions;
- zero-shot vs oracle-supplement distinction.

Do not claim:

- cross-dataset generalization before Weather / Exchange / ILI results;
- stable superiority over strongest static before the numbers support it;
- deployable cross-architecture transfer from PatchTST/iTransformer evidence.
