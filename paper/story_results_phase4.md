# Phase-4 投稿故事与结果梳理

## 1. 一句话主线

在固定 Informer 架构、训练协议、超参数和数据划分不变的情况下，仅改变前馈块中的激活函数及其放置位置，就会系统性改变长序列预测性能；激活函数不是无关紧要的默认项，而是一个强烈依赖 dataset、horizon 和 placement 的设计变量。

当前最稳妥的投稿 claim 是：

> Under a matched Informer official-script protocol with complete three-seed coverage, activation choice is a dataset- and horizon-sensitive design variable. Bounded and tanh-family activations are strong static choices on average, but the full grid rejects a universal activation or deployable factor-router claim.

中文表述：

> 在同一 Informer 协议下，激活函数选择显著依赖数据集与预测长度；bounded/tanh-family 激活在平均排名上最强，但完整矩阵同时否定了“单一激活函数通吃”和“可直接部署的因子路由器”两个过强结论。

## 2. 从草稿故事到投稿故事的收敛

草稿中的原始故事是正确的方向：时序预测研究通常重视网络结构、注意力机制、损失函数和输入窗口设计，却默认使用 GELU/ReLU/Swish 一类通用激活。我们的实验说明，在不改变网络结构的前提下，activation regime 和 placement 本身就会带来大幅性能差异。

但当前论文需要把故事收紧到三个边界：

1. 主证据仍应以 Informer full grid 为核心；PatchTST 与 iTransformer 已完成迁移验证，但结果分别是 partial/data-specific transfer 和 weak/negative direct transfer，而不是强正迁移。
2. 因子桶发现了 regime dependence，但还不能直接形成 zero-shot 选择算法。
3. 震荡激活不是 universal winner；`tanh_sin001_all` 很强，但结论应是 tanh-family/bounded family 强，而不是“震荡必然更好”。

建议论文标题/主线围绕：

- Activation-only replacement in Informer forecasting
- Dataset/horizon-sensitive activation choice
- Placement and regime diagnostics
- Full-grid evidence, not universal activation discovery

## 3. 核心假设与当前证据状态

### H1: 激活函数本身会显著影响同架构时序预测表现

状态：强支持。

证据：

- 完成 `972/972` 个 canonical seed-runs。
- `324/324` 个 dataset-horizon-config cells 都有 3 seeds。
- `34/36` 个 dataset-horizon cells 的最优配置不是 GELU。
- 所有 `20/20` 个 ETT/ETTm cells 都是非 GELU 最优。
- Solar 也有 `14/16` 个 cells 非 GELU 最优。

主文可写：

> Across 36 dataset-horizon cells and 972 matched seed-runs, non-GELU activations are best in 34 cells, showing that the activation function is a first-order design variable even when the Informer architecture and training protocol are fixed.

### H2: bounded/tanh-family 是强静态候选

状态：强支持，但不能写成 universal winner。

静态排名：

| config | mean rank | best cells | top-3 cells | mean gain vs GELU | positive cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| `tanh_sin001_all` | 2.78 | 8 | 27 | 11.7% | 83.3% |
| `tanh_all` | 3.19 | 7 | 25 | 11.9% | 80.6% |
| `softsign_all` | 3.47 | 8 | 22 | 11.2% | 80.6% |

解释：

- `tanh_sin001_all`, `tanh_all`, `softsign_all` 是最稳的三类候选。
- 它们的优势不是单点现象，而是在 36 cells 的平均排名和 top-3 覆盖中体现。
- 但 best cells 分散，说明不能只推一个函数。

### H3: 震荡扰动在高波动场景中更好

状态：部分支持，不能直接确认。

支持部分：

- `tanh_sin001_all` 是全局 mean rank 第一，best `8/36`，top-3 `27/36`。
- 在 ETTh2/24、ETTh2/168、ETTm1/96、ETTm2/96、Solar4/24、Solar6/24、Solar6/96、Solar8/24 等 cells 中最优。

不支持部分：

- 震荡并不在所有高波动或高异常 bin 中稳定占优。
- 因子桶中最强 contrast 反而经常显示 `tanh_all_outtanh` 在 high volatility/jump/turbulence bin 中显著变差。
- LOSO oracle 没有给出跨 dataset 的震荡选择规则。

投稿表述应改为：

> A small oscillatory perturbation around tanh is a strong static candidate, but the present evidence does not justify a general rule that oscillatory activations should be selected in high-volatility regimes.

### H4: placement 会显著影响表现，并且与数据集/预测长度相关

状态：强支持。

位置结果：

| placement | best cells | 解释 |
| --- | ---: | --- |
| full hidden activation | 28/36 | 默认最稳，主候选应优先放 encoder+decoder FFN |
| hidden + output tanh | 5/36 | 在部分 ETT/ETTm 长 horizon 极强，但整体平均排名差 |
| encoder-only tanh | 3/36 | 主要出现在 Solar96 |
| decoder-only tanh | 0/36 | 不建议作为主候选 |

`tanh_all_outtanh` 最优 cells：

- ETTh1/720
- ETTh2/336
- ETTh2/720
- ETTm2/288
- ETTm2/672

`enc_tanh_dec_gelu` 最优 cells：

- Solar4/96
- Solar5/96
- Solar7/96

结论：

- output tanh 不是简单的 universal clipping trick。
- 它在部分长 horizon ETT/ETTm cell 很强，但全局 mean rank 只有 7.11，mean gain 为负。
- Solar96 更倾向 encoder-side placement，而不是 output tanh。

## 4. 主要实验结果应该怎么讲

建议主结果分四层写。

### Result 1: Full-grid coverage makes the claim credible

写完整矩阵覆盖：

- 12 datasets/sites: ETTh1, ETTh2, ETTm1, ETTm2, Solar1--Solar8
- 36 dataset-horizon cells
- 9 activation configs
- 3 seeds: 2024, 2025, 2026
- 972 seed-runs
- batch size 32, 6 epochs, same protocol

目的：先建立实验不是 pilot，而是完整系统矩阵。

### Result 2: Activation-only changes beat GELU in most cells

核心数字：

- `34/36` cells 非 GELU 最优。
- ETTh/ETTm: `20/20` 非 GELU 最优。
- Solar: `14/16` 非 GELU 最优。
- Solar3/24 和 Solar3/96 是 GELU-best，作为反例证明我们没有夸大。

这比“某个激活赢”更重要：说明默认 GELU 不是稳健选择。

### Result 3: Bounded/tanh-family are the best static candidates

主表/图可以展示 static ranking：

- `tanh_sin001_all`
- `tanh_all`
- `softsign_all`

这部分支持论文的 positive claim。

### Result 4: Placement is real but not globally transferable

讲 placement：

- full-hidden 最稳。
- output tanh 在 ETT/ETTm 长 horizon 有高收益，但全局风险高。
- encoder-only tanh 在 Solar96 有信号。
- decoder-only tanh 没有赢过。

这部分是论文的 mechanistic/diagnostic contribution。

## 5. 因子桶到底发现了什么

### 已发现

Summary-level factor-bin 覆盖全矩阵：

- 36 dataset-horizon cells
- 783 factor-bin rows
- 196 个 `|end-start improvement| >= 0.02` 的 factor-conditioned contrasts

这说明：不同 regime 下，不同 activation 的相对表现确实变化。

Sample-level oracle 更严格：

- 84 个 LOSO oracle rows
- 只有 7 个 positive
- 全部来自 Solar1/96
- positive factors 包括 `max_abs_change`, `volatility`, `volatility_shock`, `trend_consistency`, `range` 等

### 不能得出的结论

不能写：

- “高波动时应该选 tanh”
- “高 volatility 应该启用震荡激活”
- “因子桶可以直接 zero-shot 选择 activation”
- “我们已经提出可部署的 activation router”

原因：

- summary-level bin 是后验 cell-level 对比，不是前瞻路由。
- sample-level LOSO 只有 Solar1/96 有正向泛化信号。
- 很多强 contrast 是某些配置在 high bin 变差，而不是给出稳定正向选择规则。

### 可以写的结论

可以写：

> Factor-bin diagnostics reveal substantial regime dependence, but the current leave-one-seed-out evidence is too weak and too localized to support a deployable activation router. We therefore treat factors as diagnostic tools for designing targeted placement ablations, not as a final zero-shot selection algorithm.

## 6. 当前可给出的保守选择规则

这不是论文主算法，只是基于当前结果的经验推荐。

### 如果只能选一个默认候选

优先试：

1. `tanh_sin001_all`
2. `tanh_all`
3. `softsign_all`

原因：三者 mean rank 和 top-3 覆盖最好。

### 如果可以按数据族选择

ETTh/ETTm：

- 默认跑 full-hidden `tanh_sin`, `tanh`, `softsign`。
- 对长 horizon，尤其 ETTh2/336/720、ETTm2/288/672，额外测试 `tanh_all_outtanh`。

Solar：

- 默认仍跑 full-hidden bounded/tanh-family。
- Solar96 额外测试 `enc_tanh_dec_gelu`。
- 不要默认 output tanh。
- Solar3 保留 GELU 作为强 baseline，因为两个 Solar3 horizons 都是 GELU-best。

### 不建议

- 不建议 decoder-only tanh，当前 best cell 为 0。
- 不建议把 ReLU 当主候选，只赢 Solar7/24 且整体 rank 弱。
- 不建议直接用 volatility/jump/turbulence 规则选择 tanh 或 output tanh。

## 7. 投稿前最值得补的实验

按性价比排序。

### E1. PatchTST 最小迁移验证

状态：已完成最小迁移验证，应作为 architecture-transfer stress test，而不是强泛化证据。

完成矩阵：

- 模型：PatchTST
- datasets: ETTh2, ETTm2, Solar2, Solar3, Solar5
- horizons: ETTh2 `168/336/720`, ETTm2 `96/288/672`, Solar `24/96`
- Stage1 configs: `gelu_ffn_linear`, `tanh_ffn_linear`, `softsign_ffn_linear`, `tanh_sin001_ffn_linear`
- Stage1 coverage: `144/144` seed-runs, all cells 3 seeds
- Stage2 placement configs: `gelu_ffn_outtanh`, `tanh_ffn_outtanh`, `tanh_sin001_ffn_outtanh`
- Stage2 coverage: `54/54` seed-runs, all cells 3 seeds

结果：

- PatchTST Stage1 非 GELU best 为 `5/12` cells，而 GELU best 为 `7/12` cells。
- 非 GELU wins 集中在 ETTh2/720、Solar2/24、Solar2/96、Solar5/24、Solar5/96。
- `softsign_ffn_linear` best `3/12`，`tanh_sin001_ffn_linear` best `2/12`，`tanh_ffn_linear` best `0/12`。
- Output-tanh placement 在 PatchTST 上没有迁移：`54/54` paired seed-runs 都输给对应 linear head，平均退化约 `5.3%--5.4%`。

结论：

PatchTST 支持“激活敏感性存在且是 architecture-dependent”的故事，但不支持“Informer 激活排序直接迁移到 PatchTST”。主文可以写 PatchTST transfer is partial，并把 output-tanh negative result 作为避免过度泛化的证据。

### E2. iTransformer 快速迁移验证

状态：已完成单 seed 快速迁移验证，应作为第二个 architecture-transfer stress test。

完成矩阵：

- 模型：iTransformer
- datasets: ETTh2, ETTm2
- horizons: `96/336/720`
- configs: `gelu_ffn_linear`, `tanh_ffn_linear`, `softsign_ffn_linear`, `tanh_sin001_ffn_linear`
- seed: `2024`
- coverage: `24/24` finite rows，其中 2 条 smoke 复用，22 条新增云端 4090 训练
- placement: 只测 FFN activation，不测 output-tanh

结果：

- GELU best `5/6` cells。
- 非 GELU best `1/6` cells。
- 唯一非 GELU best 是 ETTm2/96：`tanh_sin001_ffn_linear` MSE `0.183869`，GELU MSE `0.184226`，相对提升约 `0.194%`。
- ETTh2/96、ETTh2/336、ETTh2/720、ETTm2/336、ETTm2/720 都是 GELU-best。

结论：

iTransformer 不支持“Informer tanh-family 排名直接迁移”。它支持更保守的故事：激活敏感性不是 Informer 独有，但最佳激活族与 placement 强烈依赖架构。由于目前只有单 seed，不应写成强统计结论。

### E3. Prospective factor-router validation

目的：验证“因子桶能否指导选择”。

设计：

- 在训练 split 或一组 dataset-horizon 上学习规则。
- 在 held-out dataset/horizon 或 held-out seed 上固定规则。
- 只能用输入窗口因子，不能看测试误差后再选。
- 比较：
  - best static activation
  - GELU
  - oracle upper bound
  - learned factor router

成功标准：

- router 显著优于 best static 或至少接近 oracle，同时在多个 cells 正向。

当前证据不足以直接跳到这个 claim。

### E3. Targeted placement ablation

目的：解释 output tanh 和 encoder-only tanh 的边界。

重点 cells：

- output tanh positive: ETTh2/336, ETTh2/720, ETTm2/288, ETTm2/672
- encoder-only tanh positive: Solar4/96, Solar5/96, Solar7/96
- counterexamples: Solar3/24, Solar3/96, selected Solar cells where output tanh bad

可加配置：

- output-only tanh with GELU hidden
- hidden tanh + linear output
- hidden tanh + output tanh
- encoder-only tanh
- decoder-only tanh

### E4. Lee/oscillatory exact-protocol cleanup

目的：如果还想讲 Lee 或更复杂 oscillation，必须把它从小 batch diagnostic 变成同协议 evidence。

当前建议：不要作为主文核心，除非补齐 batch32/3 seeds/关键 cells。

## 8. 建议论文贡献写法

建议 contributions 写成：

1. We introduce an activation-only evaluation interface for Informer that changes hidden activation family and placement while holding attention, width, depth, optimizer, and training protocol fixed.
2. We provide a complete three-seed full-grid study over 36 dataset-horizon cells and 972 seed-runs, showing that non-GELU activations are best in 34/36 cells.
3. We show that bounded/tanh-family activations are strong static candidates, but no single activation universally dominates.
4. We disentangle placement effects, showing that output tanh, encoder-only tanh, and full-hidden replacement have different dataset/horizon profiles.
5. We use factor-bin and oracle diagnostics to reveal regime dependence, while showing that current evidence is insufficient for a deployable zero-shot activation router.

## 9. 论文中应避免的过强说法

避免：

- “We propose a zero-shot activation selection algorithm”。
- “High volatility should use tanh/oscillatory activation”。
- “Oscillatory activation is generally better”。
- “The Informer activation ranking transfers to PatchTST”。
- “The Informer activation ranking transfers to iTransformer”。
- “Output tanh is robust across architectures”。
- “We surpass Informer Table 2”。

替换为：

- “We provide diagnostic evidence toward activation selection.”
- “Factor bins reveal regime dependence but do not yet support deployment.”
- “The complete Informer grid identifies strong static candidates and placement-specific effects.”
- “PatchTST transfer is partial: activation sensitivity remains visible, but the best activation and placement are architecture-dependent.”
- “iTransformer is a weak/negative direct-transfer probe: GELU remains best in most tested cells, so architecture matters.”

## 10. 当前结果资产

主结果：

- `phase4_remote_results/analysis_current/phase4_protocol_aggregate.csv`
- `phase4_remote_results/analysis_current/phase4_completion_matrix.csv`
- `phase4_remote_results/analysis_current/phase4_claim_summary.md`

因子桶：

- `phase4_remote_results/analysis_current/factors/phase4_factor_bin_summary.csv`
- `phase4_remote_results/analysis_current/factors/phase4_factor_contrast_summary.csv`
- `phase4_remote_results/analysis_current/factors/phase4_factor_oracle_loso_summary.csv`
- `phase4_remote_results/analysis_current/factors/phase4_factor_claim_summary.md`

论文图表：

- `figures/TABLE_phase4_coverage.tex`
- `figures/TABLE_phase4_best_activation.tex`
- `figures/TABLE_phase4_static_ranking.tex`
- `figures/TABLE_phase4_loso_oracle.tex`
- `figures/fig_phase4_best_gain_heatmap.pdf`
- `figures/fig_phase4_static_ranking.pdf`
