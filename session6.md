# Session 6 - Phase-5 Router-Core Takeover

Date: 2026-05-31

## Objective

把 Phase-5 主线从“factor router 只是诊断工具”调整为“验证 router 作为论文核心算法的可能性”。本轮只做本地文档、分析脚本、schema/queue 准备与 CPU-only 复核；没有启动任何本地或云端深度学习训练。

## Implemented

- 扩展 `lee_ocil/scripts/phase4_pairwise_factor_router.py`：
  - 固定 Phase-5 core action space；
  - 默认只使用输入数据统计/时频因子，不把 topology 当作 zero-shot 输入；
  - 新增 `leave_cell`、`leave_dataset`、`leave_horizon`、`leave_family` split；
  - 新增 cell-level router 组合逻辑；
  - 输出 router vs `gelu_all`、router vs train-fold strongest static、regret vs oracle、oracle capture ratio、abstain rate、action/config hit rate、placement precision；
  - 新增 `--core_only`、`--scope_filter`、`--fallback_policies`、`--include_topology`。
- 给 Informer 训练入口补充第一波跨数据集名称：
  - `Weather`
  - `Exchange`
  - `ILI`
- 新增本地新数据集预检：
  - `lee_ocil/scripts/phase5_dataset_preflight.py`
- 新增第一波 oracle matrix 生成器：
  - `lee_ocil/scripts/phase5_build_router_oracle_matrix.py`
- 新增 Phase-5 Router-Core 文档工件：
  - `.aris/phase5_router_validation_plan_20260531.md`
  - `.aris/phase5_context_pack_20260531.md`

## Local Retrospective Router Result

运行命令：

```bash
python lee_ocil/scripts/phase4_pairwise_factor_router.py \
  --factors_dir .aris/phase4_timefreq_factors_20260523_router \
  --output_dir .aris/phase5_router_core_20260531 \
  --scope_filter Informer-fullgrid \
  --core_only \
  --modes leave_cell leave_dataset leave_horizon
```

主要输出：

- `.aris/phase5_router_core_20260531/phase5_router_cell_summary.csv`
- `.aris/phase5_router_core_20260531/phase5_router_cell_predictions.csv`
- `.aris/phase5_router_core_20260531/phase5_router_cell_tables.md`

Informer-only、input-factor-only 的关键结果：

| split | best feature set | mean gain vs GELU | mean gain vs train static | mean regret vs oracle | oracle capture ratio | abstain rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| leave-cell | frequency | 0.1237 | 0.00003 | 0.0472 | 0.6409 | 0.0000 |
| leave-dataset | all_input/data/statistical | 0.1236 | -0.00010 | 0.0473 | 0.6383 | 0.0000 |
| leave-horizon | all_input/data/statistical | 0.1232 | -0.00048 | 0.0477 | 0.6243 | 0.0000 |

Interpretation:

- Router 明显超过 `gelu_all`，支持把 router 作为主线继续验证。
- Router 与 train-fold strongest static 基本持平，尚不能写成“稳定超过强 static baseline”。
- Router 捕获约 62%-64% oracle gain，说明方法不是随机选择，但仍需要跨数据集和 calibrated/oracle-supplement 阶段来决定最终论文口径。

## New Dataset Status

本地预检命令：

```bash
python lee_ocil/scripts/phase5_dataset_preflight.py \
  --output_dir .aris/phase5_new_dataset_preflight_20260531 \
  --loader_smoke
```

结果：

- Weather / Exchange / ILI CSV 当前不在本地候选路径中。
- 因文件缺失，未执行 loader shape smoke。
- 报告见 `.aris/phase5_new_dataset_preflight_20260531/phase5_new_dataset_preflight.md`。

第一波 1-seed oracle matrix 已生成：

- `.aris/phase5_new_dataset_oracle_matrix_20260531.csv`
- 54 runs = 3 datasets x 3 horizons x 6 configs x seed 2024

## Current Claim Boundary

可推进的论文主线：

> A zero-shot activation router framework can be proposed and validated retrospectively on the complete Informer grid. Current evidence shows strong gains over GELU and meaningful oracle capture, but cross-dataset generalization and improvement over the strongest static baseline still require Weather / Exchange / ILI validation.

当前不能写成结论：

- zero-shot router 已经稳定超过 strongest static；
- router 已经跨数据集泛化；
- 1-seed oracle matrix 能支持统计结论；
- PatchTST/iTransformer 已证明跨架构直接迁移。

## Next Gate

1. 获取或确认云端存在 Weather / Exchange / ILI CSV。
2. 在云端先跑 `phase5_dataset_preflight.py --loader_smoke`，不训练。
3. Fresh remote preflight 后启动 54-run 1-seed oracle matrix。
4. 先做 zero-shot evaluation；若不赢 static，再转 oracle-supplement calibration，不放弃 router 主线。
