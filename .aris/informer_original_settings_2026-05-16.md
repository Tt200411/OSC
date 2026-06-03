# Informer 原始实验设定核对

日期：2026-05-16  
论文文件：`2012.07436v3 (1).pdf`  
论文：Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting", AAAI 2021  
官方代码来源：`zhouhaoyi/Informer2020`

## 1. 重要纠正

`2012.07436v3` 是 Informer 原论文。  
`2208.14111v3` 不是 Informer，而是 "Transformers with Learnable Activation Functions"。

## 2. 论文正文给出的通用训练设定

PDF 的 Experimental Details 写到：

- optimizer: Adam
- learning rate: `1e-4`
- learning-rate decay: 每个 epoch 减半
- total epochs: 8
- early stopping: proper early stopping
- batch size: 32
- loss: MSE
- input preprocessing: zero-mean normalized
- train/val/test split:
  - ETT: 12/4/4 months
  - Weather: 28/10/10 months
- Table 2 是 multivariate forecasting，即 `features=M`。

注意：官方 repo 的 `main_informer.py` 默认 `train_epochs=6`，与论文正文的 8 有差异。官方 `scripts/*.sh` 没有显式覆盖 `--train_epochs`，所以直接跑官方脚本通常会使用 repo 默认 6 epoch。

## 3. 模型组件设定

从论文 Appendix E / Table 7 和官方 `main_informer.py` 可见：

- model: `informer`
- attention: `prob`
- embedding dimension: `d_model=512`
- FFN inner dimension: `d_ff=2048`
- activation: `gelu`
- dropout:
  - 论文 Appendix Table 7: `p=0.1`
  - 官方 repo 默认: `dropout=0.05`
- official default:
  - `n_heads=8`
  - `factor=5`
  - `embed=timeF`
  - `distil=True`
  - `mix=True`
  - `batch_size=32`
  - `patience=3`
  - `loss=mse`
  - `lradj=type1`

Appendix Table 7 中 encoder block 描述为：

- embedding `d=512`
- Multi-head ProbSparse Attention
- Add + LayerNorm + Dropout
- position-wise FFN `d_inner=2048`, GELU
- Add + LayerNorm + Dropout
- distilling: `1x3 Conv1d`, ELU, max pooling stride 2

decoder block：

- masked self-attention
- multi-head attention
- FFN `d_inner=2048`, GELU
- final FCN output projection

## 4. Table 2 multivariate 脚本设定

这些来自官方 repo 脚本，并与本项目 `lee_ocil/scripts/*.sh` 保留内容一致。

### ETTh1, features=M

| Horizon | seq_len | label_len | pred_len | e_layers | d_layers | factor | itr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 48 | 48 | 24 | 2 | 1 | 3 | 5 |
| 48 | 96 | 48 | 48 | 2 | 1 | 5 | 5 |
| 168 | 168 | 168 | 168 | 2 | 1 | 5 | 5 |
| 336 | 168 | 168 | 336 | 2 | 1 | 5 | 5 |
| 720 | 336 | 336 | 720 | 2 | 1 | 5 | 5 |

### ETTh2, features=M

| Horizon | seq_len | label_len | pred_len | e_layers | d_layers | factor | itr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 48 | 48 | 24 | 2 | 1 | 5 | 5 |
| 48 | 96 | 96 | 48 | 2 | 1 | 5 | 5 |
| 168 | 336 | 336 | 168 | 3 | 2 | 5 | 5 |
| 336 | 336 | 168 | 336 | 3 | 2 | 5 | 5 |
| 720 | 720 | 336 | 720 | 3 | 2 | 5 | 5 |

### ETTm1, features=M

| Horizon | seq_len | label_len | pred_len | e_layers | d_layers | factor | itr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 672 | 96 | 24 | 2 | 1 | 5 | 5 |
| 48 | 96 | 48 | 48 | 2 | 1 | 5 | 5 |
| 96 | 384 | 384 | 96 | 2 | 1 | 5 | 5 |
| 288 | 672 | 288 | 288 | 2 | 1 | 5 | 5 |
| 672 | 672 | 384 | 672 | 2 | 1 | 5 | 5 |

### Weather/WTH, features=M

| Horizon | seq_len | label_len | pred_len | e_layers | d_layers | factor | itr |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 168 | 168 | 24 | 3 | 2 | 5 | 3 |
| 48 | 96 | 96 | 48 | 2 | 1 | 5 | 3 |
| 168 | 336 | 168 | 168 | 3 | 2 | 5 | 3 |
| 336 | 720 | 168 | 336 | 3 | 2 | 5 | 3 |
| 720 | 720 | 336 | 720 | 3 | 2 | 5 | 3 |

## 5. 和我们当前 Phase-1 设定的关键差异

当前 Phase-1 激活实验大量使用：

- `seq_len=96`
- `label_len=48`
- `e_layers=2`
- `d_layers=1`
- `train_epochs=6`

这与 Informer Table 2 的 horizon-specific 设置不同，尤其是：

- ETTh1/168 原设定是 `seq_len=168,label_len=168`，我们常用 `96/48`。
- ETTh1/336 原设定是 `seq_len=168,label_len=168`，我们常用 `96/48`。
- ETTh2/168 原设定是 `seq_len=336,label_len=336,e_layers=3,d_layers=2`，我们常用 `96/48,2/1`。
- ETTh2/336 原设定是 `seq_len=336,label_len=168,e_layers=3,d_layers=2`，我们常用 `96/48,2/1`。

因此，若要严格比较截图 Table 2，必须按上述 Table 2 脚本设定重跑 GELU/tanh/Lee/softsign，并冻结 seed 与去重策略。

