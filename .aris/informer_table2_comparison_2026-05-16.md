# Informer Table 2 对齐检查

日期：2026-05-16  
截图：`截屏2026-05-16 下午5.32.11.png`  
本地结果来源：`phase1_remote_results/all_servers_current/phase1_activation_summary.csv`

## 结论

当前结果不能整体宣称“复现或超越 Informer Table 2”。我们只在 ETTh1/ETTh2 的部分 horizon 上有可比数据；ETTm1、Weather、ECL 还没有对应结果。即便在 ETTh1/ETTh2，当前 Phase-1 脚本大多使用固定 `seq_len=96,label_len=48,e_layers=2,d_layers=1`，而原 Informer 脚本对不同 horizon 使用不同 `seq_len/label_len/e_layers/d_layers`，所以严格来说还不是同协议复现。

## 截图中的 Informer Table 2 数字

截图是 Informer 论文的 multivariate long sequence time-series forecasting 表。可和当前本地结果直接对照的数据集只有 ETTh1 和 ETTh2 的部分 horizon。

| Dataset | Horizon | Informer MSE | Informer MAE | 本地最好 config | 本地 MSE | 本地 MAE | Seeds | 是否低于截图 MSE |
|---|---:|---:|---:|---|---:|---:|---:|---|
| ETTh1 | 24 | 0.577 | 0.549 | Lee1 | 0.6832 | 0.6046 | 3 | 否 |
| ETTh1 | 168 | 0.931 | 0.752 | Lee1 | 1.0608 | 0.8248 | 6 | 否 |
| ETTh1 | 336 | 1.128 | 0.873 | Lee3 | 1.1898 | 0.8616 | 2 | MSE 否；MAE 是 |
| ETTh2 | 24 | 0.720 | 0.665 | tanh | 0.7221 | 0.6769 | 1 | 基本接近但否 |
| ETTh2 | 168 | 3.489 | 1.515 | Lee3 | 2.7308 | 1.3802 | 6 | 是 |
| ETTh2 | 336 | 2.723 | 1.340 | Lee3 | 2.1242 | 1.1048 | 1 | 是，但只有 1 seed |

## tanh 单独对照

| Dataset | Horizon | Informer MSE | tanh MSE | tanh MAE | Seeds | 判断 |
|---|---:|---:|---:|---:|---:|---|
| ETTh1 | 24 | 0.577 | 0.6869 | 0.6085 | 3 | 未达到 |
| ETTh1 | 168 | 0.931 | 1.1400 | 0.8586 | 6 | 未达到 |
| ETTh1 | 336 | 1.128 | 1.3308 | 0.9464 | 2 | 未达到 |
| ETTh2 | 24 | 0.720 | 0.7221 | 0.6769 | 1 | 基本接近但未达到 |
| ETTh2 | 168 | 3.489 | 4.0126 | 1.7203 | 6 | 未达到 |
| ETTh2 | 336 | 2.723 | 2.5250 | 1.2669 | 2 | 达到，但 seed 不足 |

## 协议差异

本地 Phase-1 激活实验一般固定：

- `seq_len=96`
- `label_len=48`
- `e_layers=2`
- `d_layers=1`
- `train_epochs=6`

原 Informer 脚本对 ETTh1/ETTh2 multivariate 使用 horizon-specific config，例如：

- ETTh1/24: `seq_len=48,label_len=48,e_layers=2,d_layers=1`
- ETTh1/168: `seq_len=168,label_len=168,e_layers=2,d_layers=1`
- ETTh1/336: `seq_len=168,label_len=168,e_layers=2,d_layers=1`
- ETTh2/168: `seq_len=336,label_len=336,e_layers=3,d_layers=2`
- ETTh2/336: `seq_len=336,label_len=168,e_layers=3,d_layers=2`

因此，当前 ETTh2/168 Lee3 的结果非常值得追，但不能直接写成“严格同协议超过 Informer 原论文”，除非重新按原脚本配置跑 GELU/tanh/Lee3 并冻结去重、统计检验。

## 对论文叙事的影响

1. “tanh 全方位领先”不是当前全局事实。tanh 在 Solar 和部分 ETT 设置中相对 GELU 很强，但在 ETTh2/168 上 Lee3 明显强于 tanh；在 ETTh1 上本地最好也没有达到截图中的 Informer MSE。
2. 学界并非完全不知道 tanh。更合理的解释是：tanh 的 bounded/zero-centered/saturation 行为是已知的，但 LTSF Transformer FFN 中的 placement + factor-conditioned 效果没有被系统研究。
3. 最值得优先追的可能不是“tanh 赢所有”，而是：
   - bounded activation 是否稳定优于 GELU；
   - encoder-side hidden tanh 是否是主要机制；
   - Lee3 在 ETTh2 long horizon 上是否能按原 Informer protocol 复现超过原论文。

