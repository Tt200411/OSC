# Session 3：Phase-2 Informer 激活函数实验与论文更新

日期：2026-05-16  
工作目录：`/Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main`

## 当前决策

本阶段不再等待低 batch Lee 队列和 Solar core 慢队列。它们继续在远端运行，但不进入当前论文主结论。当前 paper 更新采用已经完成并同步的 Phase-2 official-script protocol 快照。

当前本地分析快照：

- 原始 phase-tagged 行：251
- 去重后行：179
- 聚合行：68
- 聚合目录：`phase2_remote_results/analysis_current/`
- 主表生成脚本：`figures/gen_phase2_paper_assets.py`

## 实验协议边界

主结果只使用 batch size 32、三 seed、同 official-script protocol 的结果：

- `features=M`
- `d_model=512`
- `n_heads=8`
- `d_ff=2048`
- `attn=prob`
- `embed=timeF`
- `dropout=0.05`
- `epochs=6`
- `lr=1e-4`
- `patience=3`

ETTh1/168、ETTh1/336 使用 `seq_len=168,label_len=168,e_layers=2,d_layers=1,factor=5`。  
ETTh2/168 使用 `seq_len=336,label_len=336,e_layers=3,d_layers=2,factor=5`。  
ETTh2/336 使用 `seq_len=336,label_len=168,e_layers=3,d_layers=2,factor=5`。  
Solar96 使用 `seq_len=96,label_len=48,e_layers=2,d_layers=1,factor=5`，作为站点扩展，不作为 Informer Table 2 复现。

## 主结果

ETT batch32 三 seed 主表：

| Dataset/H | GELU | tanh | softsign | tanh_sin a=0.01 | 最好非 GELU 改善 |
|---|---:|---:|---:|---:|---:|
| ETTh1/168 | 1.047±0.021 | **0.912±0.131** | 0.933±0.147 | 0.921±0.138 | +12.9% |
| ETTh1/336 | 1.361±0.026 | 1.230±0.076 | **1.219±0.042** | 1.227±0.055 | +10.4% |
| ETTh2/168 | 5.142±1.165 | 3.078±0.489 | 3.145±0.537 | **2.976±0.366** | +40.4% |
| ETTh2/336 | 3.928±0.187 | 2.178±0.228 | 2.314±0.202 | **2.172±0.179** | +44.6% |

最强可写结论：

> 在 Informer official-script protocol 下，bounded hidden activations 及 tanh-family hidden variants 在 ETTh1/ETTh2 长 horizon 上系统性优于同环境 GELU baseline；ETTh2 上优势尤其大。

不能写：

- “复现并全面超过 Informer 原论文。”
- “tanh 全方位领先。”
- “Lee1/Lee3 稳定优于 tanh。”
- “output tanh 普遍有效。”
- “高波动序列一定更适合 tanh 或 oscillatory activation。”

## 位置实验

当前 placement 结论：

- ETTh1：hidden tanh 比 output tanh 更可信；output tanh 在 168 和 336 上都不如 linear output。
- ETTh2：output tanh 在 336 上最强，在 168 上也有竞争力；这是 dataset/horizon-specific 的 output range bounding 效应。
- Solar96：output tanh 明显有害；Solar1/96 更偏 encoder-side tanh，Solar5/96 更偏 hidden boundedness。
- decoder-only tanh 整体偏弱。

因此论文必须把 hidden activation 和 output activation 分开讨论，不能混成同一个 “tanh 机制”。

## Factor-bin 状态

严格 Solar96 数组已经完成并分析：

- 结果目录：`phase2_remote_results/10.21.53.82/analysis_factor_solar96_phase2/`
- 样本行：334080
- bin summary：213 行
- contrast summary：72 行

初步结论：

- factor-bin 信号存在，但不是单调普遍规律。
- Solar1/96 中，tanh vs GELU 在 high-mean 桶更强；tanh_sin vs tanh 在部分 high max-change 桶更好，但在 high std 桶更差。
- Solar5/96 中，tanh vs GELU 反而在 low max-change / low volatility 桶更强。

论文中只能写成“factor-bin 是诊断性、条件相关的证据”，不能写成机制已被普遍验证。

## Lee 状态

低 batch Lee 队列仍在远端继续跑，但当前 paper 不等待。

已同步可用诊断：

- ETTh1/168 batch8：tanh `0.947±0.057` 最好；Lee1 `0.979±0.010` 优于 GELU 但不优于 tanh；Lee3 `1.091±0.086` 不稳定。
- ETTh2/168 batch4：Lee1/Lee3 有强单 seed 信号，但 batch 与 seed 都不足，只能作为 appendix pilot。

## Paper 更新

已经更新：

- `paper/main.tex`
- `paper/sections/0_abstract.tex`
- `paper/sections/1_introduction.tex`
- `paper/sections/3_method.tex`
- `paper/sections/4_experiments.tex`
- `paper/sections/5_conclusion.tex`
- `paper/sections/A_appendix.tex`
- `figures/gen_phase2_paper_assets.py`
- `paper/figures/TABLE_phase2_main_ett.tex`
- `paper/figures/TABLE_phase2_placement.tex`
- `paper/figures/TABLE_phase2_lee_pilot.tex`
- `paper/figures/fig_phase2_relative_gains.pdf`

论文叙事已经从 Phase-1 “oscillatory/Lee 候选” 改为 Phase-2 “bounded hidden activation under official-script protocol”。
