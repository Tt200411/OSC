# Factor-Guided Activation Routing for Long-Term Time-Series Forecasting

本仓库是一个长周期时间序列预测研究项目，核心问题是：在 Transformer-style forecasting 模型中，FFN 内部的激活函数是否真的是一个可以默认忽略的实现细节？当前版本的论文主线正在从单一震荡激活函数扩展为 **factor-guided activation regime / placement router**。

## 1. 研究动机

长周期时间序列预测的主流工作通常把创新集中在网络结构和损失函数上，例如分块机制、时序分解、跨通道建模、注意力近似或更复杂的预测头。与此相对，前馈网络中的非线性激活函数大多沿用 `GELU` 或 ReLU-family 默认设置。

本项目的实验表明，这个默认值并不总是无害的。在相同数据集、相同 horizon、相同模型结构、相同优化协议下，仅改变 activation regime 或 activation placement，就可能显著改变预测误差和泛化表现。

## 2. 论文故事

论文草稿按以下逻辑组织：

1. **默认 GELU 留下了性能空白。**  
   现有长周期预测模型很少系统研究 FFN 非线性，但激活函数控制饱和、符号保持、梯度尺度和表示范围。

2. **激活函数和放置位置是条件选择问题。**  
   最佳配置不是一个全局常数，而是由数据集特征、预测 horizon、模型拓扑结构和 placement 共同决定。

3. **因子桶可以提供低成本选择信号。**  
   项目借鉴金融预测中的因子分析，把输入窗口的波动性、频谱、跳变、趋势、熵和跨通道统计转成 factor buckets，用于解释和尝试选择 activation regime / placement。

4. **router 是当前 Phase-5 的核心算法方向。**  
   当前路线不是直接预测 best config，而是学习 pairwise uplift + abstain 规则：当输入因子触发某个 action 时启用对应激活配置；低置信度时回退到 `gelu_all` 或训练集强 static baseline。

## 3. 核心假设

本文围绕三个假设展开。

- **H1: 震荡微扰增强表达。**  
  在有界非线性上加入小幅震荡，例如 `tanh_sin001`，可能在高波动或高频模式下提高表达能力。

- **H2: 架构与激活形式敏感性。**  
  `tanh`、`softsign`、`tanh_sin`、`ReLU`、`GELU` 等激活形式在不同模型拓扑中并不等价，Informer、PatchTST 和 iTransformer 上的排序不能直接互相转移。

- **H3: placement 与数据特质强耦合。**  
  encoder-side、decoder-side、full-hidden、output-bounded placement 会产生不同效果，且最佳 placement 与 horizon、频谱结构、波动性和数据集形态相关。

## 4. 当前方法

当前方法包含两层。

### Activation Search Space

主要候选包括：

- Baseline: `gelu_all`
- Regime: `tanh_all`, `softsign_all`, `tanh_sin001_all`
- Placement: `enc_tanh_dec_gelu`, `enc_gelu_dec_tanh`, `tanh_all_outtanh`

### Factor Router

router 使用输入窗口统计因子学习 pairwise action uplift：

- 不直接预测唯一 best config
- 对每个 action 学习是否启用
- 多个 action 触发时按训练 fold 的 validation uplift 排序
- 低置信度时 abstain 到 `gelu_all` 或强 static baseline

当前 Phase-5 结果显示，**pure zero-shot router 还不可靠**；更稳妥的论文路线是 **oracle-supplemented / calibrated factor router**。

## 5. 实验状态

### Informer Full Grid

Informer 是当前主证据：

- 36 dataset-horizon cells
- 9 activation configurations
- 3 seeds per configuration
- 972 / 972 matched seed-runs complete
- 324 / 324 dataset-horizon-configuration cells complete

结论边界：

- 非 GELU 激活在大多数 Informer cells 中优于默认 GELU。
- bounded / tanh-family 激活是强 static candidates。
- 但不存在一个可以无条件替代 GELU 的 universal activation。

### Cross-Dataset Router Stress Test

Phase-5 新增了 `Weather`, `Exchange`, `ILI`：

- 9 dataset-horizon cells
- 7 activation / placement configs
- 3 seeds
- 189 / 189 seed-runs complete

主要结果：

- oracle search space 在 9 / 9 cells 中优于 `gelu_all`
- 平均 oracle gain vs GELU 约为 `+2.40%`
- pure zero-shot old-grid router 失败，平均 gain vs GELU 约为 `-11.12%`
- oracle-supplement selection 有弱正信号，held-out seeds 上整体 gain vs GELU 约为 `+0.69%`
- calibrated split 仍不能稳定赢 GELU，但能减少部分 static baseline 错误

因此当前支持的 claim 是：

> Activation regime and placement are dataset/horizon-sensitive. A small oracle calibration matrix can support a low-cost calibrated activation/placement selector, but a pure zero-shot cross-dataset router is not yet reliable.

## 6. 论文 Draft

论文 draft 位于 `paper/`：

- `paper/main.tex`: 当前 LaTeX 主稿
- `paper/sections/`: 分章节草稿
- `paper/draft.md`: 早期中文故事草稿
- `paper/投稿故事.md`: 当前投稿叙事线索
- `paper/submission_hypothesis_full.tex`: 投稿假设扩展稿
- `paper/main.pdf`: 已编译 PDF 版本

目前 draft 的证据边界需要保持保守：

- Informer full-grid 是完整、多 seed 的主证据。
- PatchTST / iTransformer 是架构迁移压力测试，不是 universal transfer claim。
- Phase-5 router 是核心算法方向，但应写成 calibrated / oracle-supplemented router，而不是已经成熟的 deployable zero-shot selector。

## 7. 重要代码与产物

核心代码：

- `lee_ocil/phase4_informer.py`
- `lee_ocil/exp/exp_informer.py`
- `lee_ocil/scripts/phase4_pairwise_factor_router.py`
- `lee_ocil/scripts/phase5_zero_shot_router_predict.py`
- `lee_ocil/scripts/phase5_calibrated_router_eval.py`
- `lee_ocil/scripts/phase5_oracle_supplement_selection.py`
- `lee_ocil/scripts/phase5_factor_signal_audit.py`

关键实验记录：

- `.aris/phase4_latest.json`
- `.aris/phase5_router_core_decision_20260601.md`
- `.aris/phase5_router_oracle_eval_20260601_all3seeds_zero/`
- `.aris/phase5_oracle_supplement_selection_20260601_all3seeds/`
- `.aris/phase5_factor_signal_audit_20260601_granfix/`
- `.aris/external_worktree_deltas_20260603/`: PatchTST / iTransformer 外部克隆仓库的当前本地改动归档，避免在主仓提交坏 submodule。

## 8. 数据集

当前项目使用：

- ETT: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`
- Solar: `Solar1` to `Solar8`
- Cross-dataset router stress test: `Weather`, `Exchange`, `ILI`

`Weather`, `Exchange`, `ILI` 来自常用 LTSF benchmark 数据集，并已接入本仓库数据加载和 Phase-5 oracle matrix。

## 9. 运行与验证

本机只用于 CPU-only 检查、聚合分析和论文编译。深度学习训练默认在远端 4090 机器上执行。

常用本地检查：

```bash
python -m py_compile lee_ocil/scripts/phase4_summary_factor_bins.py
python -m py_compile lee_ocil/scripts/phase5_factor_signal_audit.py
python lee_ocil/scripts/phase5_factor_signal_audit.py \
  --aggregate .aris/phase5_router_oracle_eval_20260601_all3seeds_zero/phase5_oracle_aggregate.csv \
  --factors_dir .aris/phase5_timefreq_factors_20260601_router \
  --lee_root lee_ocil \
  --output_dir .aris/phase5_factor_signal_audit_20260601_granfix
```

论文编译优先使用：

```bash
cd paper
tectonic main.tex
```

## 10. 当前结论

本项目当前最稳妥的论文版本不是宣称“某个新激活函数通吃所有数据集”，而是：

> 在长周期时间序列预测中，activation regime 和 placement 是被长期忽视的条件设计变量。Informer full-grid 证明了默认 GELU 并非无害；因子桶分析和 Phase-5 stress test 显示，router 有作为低成本 calibrated activation selector 的潜力，但 pure zero-shot selector 仍需要更多跨数据集证据和更强 OOD / abstain 机制。
