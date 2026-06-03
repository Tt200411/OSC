# Phase-2 Informer 原协议激活实验阶段结论

更新时间：2026-05-16 19:25 CST

## 1. 协议边界

这里的 “strict protocol” 指 Informer 官方脚本风格协议，而不是旧 Phase-1 固定 `seq_len=96,label_len=48,e_layers=2,d_layers=1` 协议。

固定公共设置：

- `model=informer`
- `features=M`
- `d_model=512,n_heads=8,d_ff=2048`
- `attn=prob,embed=timeF`
- `batch_size=32`
- `train_epochs=6,dropout=0.05,patience=3`

说明：原论文正文有 `epochs=8, dropout=0.1` 的表述，但官方脚本/默认实现更接近 `6/0.05`。当前所有 Table 2 对齐结论都应标注为 “official-script protocol”。

## 2. 原始 GELU 是否复现 Informer Table 2

当前不能宣称已经复现 Informer 原始结果。GELU baseline 在四个已完成 horizon 上均弱于 Table 2。

| Dataset | pred_len | GELU MSE | Informer Table 2 MSE | 判断 |
|---|---:|---:|---:|---|
| ETTh1 | 168 | 1.047 | 0.931 | 未复现 |
| ETTh1 | 336 | 1.361 | 1.128 | 未复现 |
| ETTh2 | 168 | 4.863 | 3.489 | 未复现 |
| ETTh2 | 336 | 3.977 | 2.723 | 未复现 |

因此论文里不能写 “我们复现并全面超过 Informer”。更稳妥的说法是：在一个官方脚本风格的同协议复跑环境中，替换 hidden activation 后显著优于同环境 GELU baseline；其中 ETTh2 的 tanh/softsign 结果已经低于 Table 2 数值，但这不等价于完整复现原论文。

## 3. strict batch32 下 tanh/softsign 对 GELU 的结果

| Dataset | pred_len | GELU | tanh hidden + linear output | softsign hidden + linear output | 结论 |
|---|---:|---:|---:|---:|---|
| ETTh1 | 168 | 1.047 +/- 0.021 | 0.912 +/- 0.131 | 0.933 +/- 0.147 | tanh 均值最好，2/3 seed 赢 GELU |
| ETTh1 | 336 | 1.361 +/- 0.026 | 1.230 +/- 0.076 | 1.219 +/- 0.042 | tanh/softsign 3/3 seed 赢 GELU |
| ETTh2 | 168 | 4.863 +/- 1.081 | 3.185 +/- 0.290 | 3.228 +/- 0.776 | tanh/softsign 3/3 seed 赢 GELU，且低于 Table 2 |
| ETTh2 | 336 | 3.977 +/- 0.030 | 2.213 +/- 0.209 | 2.314 +/- 0.202 | tanh/softsign 3/3 seed 赢 GELU，且低于 Table 2 |

当前最强的可写 claim 不是 “tanh 全方位领先”，而是：

> 在 Informer official-script protocol 下，bounded hidden activations（tanh/softsign）在 ETTh1/ETTh2 long-horizon multivariate forecasting 中系统性优于 GELU baseline；ETTh2 上提升尤其大。

## 4. 激活函数位置效应

已完成/进行中的位置实验显示：hidden activation 与 output activation 必须拆开讨论。

ETTh1/168 三 seed：

- `tanh hidden + linear output`: 0.912，当前最好。
- `encoder tanh / decoder GELU / linear output`: 1.036，均值只略优于 GELU，且不稳定。
- `encoder GELU / decoder tanh / linear output`: 1.056，均值略差于 GELU。
- `tanh hidden + tanh output`: 1.106，明显差于 hidden-only tanh。
- `GELU hidden + tanh output`: 1.180，两 seed，偏差且不稳定。

Solar1/96 三 seed：

- `encoder tanh / decoder GELU / linear output`: 0.2246，当前最好。
- `tanh hidden + linear output`: 0.2280。
- `GELU`: 0.2330。
- `tanh hidden + tanh output`: 0.4052，灾难性变差。

Solar5/96 当前 seed2024：

- `encoder tanh / decoder GELU`: 0.1794。
- `tanh hidden`: 0.1801。
- `GELU`: 0.1818。

早期解释：Solar 上 output tanh 基本可以判为有害；encoder-side bounded hidden activation 可能更关键。ETT 上 strict protocol 目前也没有支持 output tanh 是主要收益来源。

## 5. Lee1/Lee3 状态

batch32 strict protocol 下 Lee 在 ETTh1/ETTh2 触发过 OOM，不能把失败前旧结果直接和 batch32 GELU/tanh 混比。

当前已启动 matched batch-size 队列：

- `10.21.53.62`: ETTh1/168/336, `batch_size=8`, GELU/tanh/Lee1/Lee3 或可运行子集。
- `10.21.53.113`: ETTh2/168/336, `batch_size=8`, GELU/tanh/softsign/Lee1/Lee3，三 seed。

这些结果只能在 batch8 内部比较；如果 Lee1/Lee3 胜出，可以写为 “memory-adjusted matched-batch Lee probe”，不能直接替代 strict batch32 主表。

## 6. factor-bin 结论边界

独立复核结论：factor-bin 信号明确存在，但不是全局定律。

稳定信号：

- ETTh1/24 上，`lee` vs `tanh` 的分桶效应最清楚。
- 例如低 CV/低 volatility 更适合 Lee，高 mean/high trend 桶也显示优势。
- Solar 上也有分桶效应，但强依赖站点、horizon、activation 和 baseline 配对。

不能写：

- “高波动总是更适合某激活”。
- “factor-bin 机制已经被普遍证明”。
- “tanh_sin/Lee 在所有高波动时间序列上都有优势”。

建议写法：

> Activation advantages are condition-dependent rather than uniform. Factor-bin analysis shows clear regime-specific effects on selected dataset-horizon-activation pairs, with the most stable evidence on ETTh1-24 and site/horizon-specific patterns on Solar.

## 7. 当前正在跑的任务

- `10.21.53.62`: `phase2_etth1_lee_bs8_20260516_191820`，ETTh1 Lee batch8，正在运行。
- `10.21.53.113`: `phase2_etth2_lee_bs8_20260516_191855`，ETTh2 Lee batch8，正在运行。
- `10.21.53.142`: `phase2_ett_place_20260516_182720`，ETTh1 placement，正在 ETTh1/336。
- `10.21.53.162`: `phase2_solar_act_fix_20260516_190650`，Solar activation/Lee，正在运行。
- `10.21.53.82`: `phase2_solar_place_fix_20260516_190650`，Solar placement，正在 Solar5/96。

下一步优先级：

1. 抓回并分析 Lee batch8 结果，先判断 Lee1/Lee3 是否仍有实际价值。
2. 补齐 ETTh2 placement，尤其 encoder-only tanh 与 output tanh 的拆分。
3. 继续 Solar1/Solar5 的 96 horizon placement，验证 encoder-side bounded activation 是否稳定。
4. 冻结 factor-bin 分析脚本和去重策略，避免重复 rerun 污染 claim。
