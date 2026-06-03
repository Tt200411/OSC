# 文献备忘录：激活函数位置、数据分布因子与长序列预测中的振荡扰动

日期：2026-05-16  
项目：Informer 长序列预测中的 bounded / oscillatory activation 研究  
目标定位：优先打磨到 CCF-C；若要冲 CCF-B，需要补充更多数据集、现代模型泛化和统计检验。

## 0. 核心判断

当前文献支持一个谨慎但有潜力的论文定位：

1. 激活函数不是简单的实现细节。已有工作从优化、信号传播、表达能力、Transformer FFN 设计、可学习激活等角度证明，activation choice 会显著影响训练动态和模型性能。
2. 周期或振荡激活已有文献基础，例如 SIREN、Fourier features、Snake，但这些工作主要面向隐式神经表示、坐标网络或周期函数外推，不是长序列时间序列预测 Transformer 的 FFN 激活位置问题。
3. Transformer 中的 FFN 激活也已有工作，例如 ReLU、GELU、GLU/SwiGLU、learnable rational activation，但这些工作主要在 NLP 或通用 Transformer 上讨论，不关心输入窗口的 volatility/turbulence，也没有系统拆分 encoder、decoder、output head。
4. LTSF 文献主要比较 attention、decomposition、patching、inverted dimensions、linear baselines 等结构变量，通常不把 FFN activation 当作核心机制变量。
5. 因此，本项目不宜包装成“发明一个新激活函数”。更稳的新颖点应收窄为：
   - activation placement 是 LTSF Transformer 中被低估的机制变量；
   - bounded hidden activation、oscillatory perturbation、output bounding 是不同机制；
   - 轻微振荡扰动是否有效，可能取决于数据状态，尤其是 input-window volatility/turbulence 与 horizon 的交互。

目前没有看到完全相同的工作：`factor-conditioned activation placement and oscillatory perturbation for Informer/LTSF`。但相邻文献很多，所以论文必须克制 claim，避免被审稿人归类为普通 activation function paper。

## 1. 检索记录

使用来源：

- arXiv API：检索 activation functions、periodic activations、Transformer FFN activations、LTSF Transformer、volatility time-series forecasting。
- Semantic Scholar：尝试检索已发表会议/期刊元数据；部分请求触发 HTTP 429 rate limit，因此只作为部分补充，不作为完整引用统计。
- Web / primary pages：arXiv、NeurIPS、ICLR/OpenReview、AAAI、ICML、PMLR、CVF/ECCV、ACL Anthology、IEEE/DOAJ/MDPI/Springer 等页面。

检索限制：

- Semantic Scholar 无 API key 时限流较严重，不能视为穷尽搜索。
- 关于 “tanh 是否在高波动时间序列中明显更好” 的证据非常分散，很多来自金融预测应用论文，实验设计与我们的 Informer/LTSF 设置差异较大。

## 2. 相关工作分桶

### 2.1 激活函数作为优化、表示与信号传播机制

代表性文献：

- Nair and Hinton, "Rectified Linear Units Improve Restricted Boltzmann Machines", ICML 2010。ReLU 是非饱和、稀疏激活的经典基线。
- Hendrycks and Gimpel, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415。GELU 用平滑方式加权输入，后来成为 Transformer FFN 常见默认选择。
- Ramachandran, Zoph, and Le, "Searching for Activation Functions", arXiv:1710.05941。Swish 说明小的函数形状变化可以影响性能，但收益通常依赖架构和任务。
- Klambauer et al., "Self-Normalizing Neural Networks", NeurIPS 2017。SELU 把激活函数形状与均值/方差传播、可训练深度联系起来。
- Schoenholz et al., "Deep Information Propagation", ICLR 2017。用 mean-field 分析说明激活函数和初始化共同决定信号传播、梯度衰减/爆炸以及 trainable depth。
- He et al., "Delving Deep into Rectifiers", ICCV 2015。PReLU 说明可学习激活参数可以改善表达能力。
- Chelly et al., "Trainable Highly-expressive Activation Functions", ECCV 2024。DiTAC 强调不同层可能需要不同激活形状。
- Alexandridis et al., "Adaptive Parametric Activation", arXiv:2407.08567。APA 明确提出激活函数应与数据分布匹配，这一点和我们的“因子条件有效性”有思想联系。

对我们的启示：

- 这组文献可以用来论证 activation 是真实机制变量。
- 但 “activation matters” 本身不是新颖点。我们的新颖点应是：在 LTSF Transformer 中控制 activation placement，并把效果和输入窗口数据状态因子联系起来。
- APA 可作为 “activation 与数据分布有关” 的近邻文献，但它主要研究分类/视觉/多任务分布，不是时间序列窗口 volatility/turbulence。

### 2.2 激活函数位置、层间异质性与输出单元

代表性文献：

- He et al., "Identity Mappings in Deep Residual Networks", ECCV 2016。pre-activation ResNet 证明非线性、归一化和 skip connection 的相对位置会影响优化。
- Fang et al., "Transformers with Learnable Activation Functions", Findings of EACL 2023 / arXiv:2208.14111。该工作在 Transformer 中使用 rational activation，发现 learned activation shapes 会随层变化。
- Chelly et al., ECCV 2024，也强调不同层对激活函数形状有不同需求。
- LeCun, Bottou, Orr, and Muller, "Efficient BackProp", 1998。和输入/输出缩放、激活函数选择、训练稳定性有关。
- Goodfellow, Bengio, and Courville, Deep Learning, 2016，第 6 章。输出单元选择与目标变量分布和损失函数耦合；连续无界回归通常使用 linear output。

对我们的启示：

- 必须拆分 hidden activation 和 output activation。否则审稿人会质疑 tanh 的收益来自输出范围压缩或 target normalization artifact。
- 当前代码证据对我们有利：`lee_ocil/models/model.py` 默认 `self.projection = nn.Linear(...)`，输出头是线性的；`--output_activation tanh` 只是后续诊断反事实。
- 论文中应明确区分：
  - encoder FFN activation；
  - decoder FFN activation；
  - final output activation。

### 2.3 周期/振荡激活

代表性文献：

- Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions", NeurIPS 2020。SIREN 用 sine activation 表示细节丰富的连续信号和导数。
- Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains", NeurIPS 2020。Fourier features 用于缓解 spectral bias、学习高频函数。
- Ziyin, Hartwig, and Ueda, "Neural Networks Fail to Learn Periodic Functions and How to Fix It", NeurIPS 2020 / arXiv:2006.08195。Snake 激活引入周期 inductive bias，并在温度和金融时间序列预测上做实验。
- Benbarka et al., "Seeing Implicit Neural Representations as Fourier Series", WACV 2022。分析 Fourier mapping 与 SIREN-like 表示之间的关系。
- FINER++ 等近期 INR 工作进一步扩展 variable-periodic activation，但主要还是坐标表示/图像/NeRF 等隐式表示场景。

对我们的启示：

- 这些工作支持“周期成分可能有用”的直觉，但不能直接证明 `tanh + a sin(x)` 在 Informer FFN 中对 LTSF 有用。
- Snake 与我们的思路接近，但不是同一问题。Snake 是带周期 inductive bias 的激活设计；我们的 `tanh + a sin(x)` 是围绕 bounded sign-preserving base 的小幅扰动。
- 当前实验已经显示 `tanh_sin` 相对 `tanh` 并不全局稳定，所以论文应写成 “conditional oscillatory perturbation”，不能写成 “oscillatory activation improves forecasting”。

### 2.4 Transformer FFN 激活

代表性文献：

- Vaswani et al., "Attention Is All You Need", NeurIPS 2017。原始 Transformer FFN 使用 ReLU。
- Hendrycks and Gimpel 的 GELU 后来成为 BERT 类 Transformer FFN 的常见默认选择。
- Dauphin et al., "Language Modeling with Gated Convolutional Networks", ICML 2017。GLU 体现了 gating nonlinearities 在序列模型中的价值。
- Shazeer, "GLU Variants Improve Transformer", arXiv:2002.05202。在 Transformer FFN 中比较 GLU variants，发现部分变体优于 ReLU/GELU。
- Fang et al., "Transformers with Learnable Activation Functions", Findings of EACL 2023。说明 Transformer 中固定 activation 的选择仍然值得研究，且可学习 activation 形状随层变化。

对我们的启示：

- Transformer FFN activation 是有文献支撑的设计轴。
- 但这也提高了 CCF-B 门槛：如果我们声称一般 Transformer activation 机制，审稿人可能希望看到 GLU/SwiGLU/GEGLU 或 learnable activation 控制。
- 对 CCF-C，可先把 GLU variants 放入相关工作和未来工作，因为当前论文重点是 controlled Informer + factor-conditioned analysis。

### 2.5 长序列时间序列预测背景

代表性文献：

- Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting", AAAI 2021。我们的控制骨干模型。
- Wu et al., "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting", NeurIPS 2021。强调 decomposition 和 auto-correlation。
- Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting", ICML 2022。frequency/decomposition 视角与我们的周期/因子直觉相关。
- Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023 / arXiv:2205.13504。DLinear/LTSF-Linear 提醒我们：只做 Informer 的结果不能夸成整个 LTSF 领域结论。
- Nie et al., "A Time Series is Worth 64 Words", ICLR 2023。PatchTST 是强现代 Transformer-style LTSF baseline。
- Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting", ICLR 2024。强现代 LTSF Transformer 变体。
- Chen et al., "TimeCatcher", arXiv:2601.20448。近期 volatility-aware forecasting 预印本，强调高波动/突变场景仍是 LTSF 难点。

对我们的启示：

- CCF-C 可以定位为 Informer 上的 controlled diagnostic study，但必须诚实处理去重、seed、统计检验。
- CCF-B 需要至少一个现代模型泛化：PatchTST、iTransformer 或 DLinear。PatchTST/iTransformer 检验 Transformer FFN activation 是否泛化；DLinear 则作为没有 Transformer FFN 的边界对照。

## 3. 我们和已有工作的差异

已有工作覆盖：

- 通用激活函数发明/搜索：GELU、Swish、SELU、PReLU、DiTAC、APA。
- 周期激活/周期特征：SIREN、Fourier features、Snake。
- Transformer FFN activation：GLU variants、learnable rational activations。
- LTSF 架构：Informer、Autoformer、FEDformer、DLinear、PatchTST、iTransformer。
- 部分工作把 activation 与数据分布联系起来，但多在分类、视觉或通用 representation learning 中。

目前看缺少：

- 在 LTSF Transformer 中系统拆分 encoder FFN、decoder FFN、output head activation 的 controlled study。
- 以 sample/window 为单位，把 activation 效果与 input-window volatility/turbulence 绑定。
- 在 LTSF 中直接比较 mild oscillatory perturbation 与其 bounded baseline，例如 `tanh_sin` vs `tanh`。
- 使用 interaction analysis，例如：
  `improvement ~ activation + factor_bin + activation:factor_bin + dataset + horizon`。

推荐新颖性表述：

> 我们不把振荡激活作为通用替代品，而是把 activation placement 作为长序列预测 Transformer 中的受控机制变量，并证明 bounded 与 oscillatory effects 必须结合 dataset、horizon 和 input-window factor 来解释。

避免写法：

- “我们发明了新的激活函数。”
- “振荡激活普遍提升长序列预测。”
- “Lee 总是优于 tanh。”
- “高波动导致提升。”  
  更稳的措辞是 “高波动窗口与更大的相对收益相关” 或 “high-volatility interaction signal”。

## 4. 关于 tanh 是否已被证明在高波动时间序列中明显占优

简短结论：没有看到强直接证据。

更具体地说：

1. 有研究比较了金融/高波动时间序列中的激活函数，但结果混合，并不支持 “tanh 在高波动数据中明显占优” 这个强命题。
2. 有些高波动金融预测论文反而报告 ReLU 更好。例如一个 LSTM 高波动股票峰值预测研究比较 ReLU、ELU、Sigmoid、Tanh，结果是 ReLU 最优。
3. 一篇 Expert Systems with Applications 的股票指数预测研究在 34 个市场、多种时间尺度下比较 DNN，检索摘要显示 ReLU 优于 tanh。
4. 也存在局部 tanh 正例。例如股票和加密货币 encoder-decoder LSTM/GRU 研究中，S&P 指数的 AE-GRU 使用 `tanh` 时最优，但个股和 Bitcoin 场景仍多为 ReLU/ELU 更好。这说明 tanh 可能是 dataset/model-specific，不是高波动普遍规律。
5. 还有一些 volatility-aware activation 或 volatility-aware forecasting 工作，说明“波动性要被模型显式处理”，但它们通常不是在证明 tanh 的优势，而是提出新的 volatility-aware mechanism。
6. LSTM 内部常用 tanh 控制 cell candidate/state，这只能说明 bounded zero-centered activation 对递归状态有工程价值，不能直接推出 Informer FFN 中 tanh 在高波动窗口更优。

对我们的论文意义：

- 如果我们后续用因子分桶证明 tanh 或 `tanh_sin` 在 high-volatility/high-turbulence bins 中有更大收益，这会是比较有新意的 empirical finding。
- 但当前写作不能说“已有研究证明 tanh 适合高波动时间序列”。更稳的写法是：
  > 既有金融和 volatility forecasting 文献说明高波动时间序列需要特殊建模，但关于 tanh 是否在高波动窗口中系统占优并无一致结论。因此，我们把 tanh 的收益视为一个需要通过 factor-conditioned counterfactual 验证的经验假设。

可引用的近邻证据：

- Kayim and Yilmaz, "Time Series Forecasting With Volatility Activation Function", IEEE Access 2022。提出 volatility activation function，说明激活函数可围绕时间序列波动性设计，但不是 tanh 优势证据。
- Gomes, Ludermir, and Lima, "Comparison of new activation functions in neural network for forecasting financial time series", Neural Computing and Applications 2011。较早系统比较金融时间序列中的激活函数，说明 activation choice 会影响金融预测，但不是 high-volatility tanh 定律。
- Nugraha et al., "Activation Function Sensitivity in LSTM-Based Peak Stock Price Forecasting for High-Volatility Financial Time Series", SISINFO 2026。明确是 high-volatility financial time series，但结果 ReLU 最优。
- A stock/cryptocurrency encoder-decoder LSTM/GRU study in Journal of Risk and Financial Management 2024。S&P index 的 AE-GRU 最优 activation 为 tanh，但个股/Bitcoin 场景并非 tanh 普遍占优。
- TimeCatcher, arXiv:2601.20448。近期 volatility-aware LTSF 预印本，说明高波动/突变窗口是难点，但它支持的是 volatility-aware modeling，不是 tanh-specific claim。

## 5. CCF-C 需要补什么

最低可信 CCF-C 实验包：

1. 数据集：
   - ETTh1、ETTh2、Solar1、Solar5。
   - 如果算力允许，补 Weather 或 ECL 更稳；但若 factor story 足够扎实，CCF-C 可以先以四个数据集为核心。
2. horizon：
   - 每个数据集至少 2-3 个 horizon。
   - 不要把单个 `pred_len=24` 结果称为 long-horizon result。
3. seeds：
   - 每个 dataset/horizon/config 至少 3 个 unique seeds。
   - 冻结去重规则，例如 unique `(dataset, horizon, seed, activation_signature, output_activation, run_id policy)`。
4. activation/config：
   - GELU、ReLU、tanh、softsign。
   - `tanh_sin` 至少覆盖 `a=0.005/0.01/0.05`。
   - Lee1/Lee3 作为小规模 mechanism probe 即可，不要作为默认大规模主线。
   - 必须包含 encoder-only tanh、decoder-only tanh、hidden tanh + linear output、hidden tanh + tanh output。
5. 统计：
   - paired seed-level delta against GELU；
   - paired seed-level delta against tanh；
   - confidence intervals 或 paired tests；
   - win rate 只能和 mean delta/uncertainty 一起报告。
6. 因子分析：
   - 计算 sample/window 级 `delta_mse = mse_activation - mse_baseline` 或 improvement ratio；
   - 按 input-window volatility/turbulence 分桶；
   - 报告 high-minus-low gradient；
   - 做 interaction regression：
     `improvement ~ activation + factor_bin + activation:factor_bin + dataset + horizon`。
7. 论文叙事：
   - 主 claim：bounded activations 是强 control；
   - oscillatory perturbation 是 conditional candidate；
   - factor correlation 是解释性诊断，不是因果证明。

CCF-C 的主要拒稿风险：

- 没有 frozen de-duplication policy。
- 没有 uncertainty 或 paired tests。
- claim 仍然写得像 universal improvement。
- factor table 只是 server-slice artifact，没有 pooled/frozen analysis。

## 6. CCF-B 需要补什么

CCF-B 需要更一般、更抗审稿的故事：

1. 更多数据集：
   - ETTm1、ETTm2、ECL、Weather。
   - Solar1/Solar5 保留，因为 solar volatility 是我们的核心假设场景。
2. 更多模型：
   - PatchTST 或 iTransformer：检验 Transformer FFN activation effect 是否泛化。
   - DLinear：作为没有 Transformer FFN activation 的边界对照。
   - Autoformer/FEDformer 可作为可选扩展，前提是集成成本可控。
3. 更多 activation controls：
   - GLU/GEGLU/SwiGLU 作为现代 Transformer FFN control。
   - learnable/parameterized activation 作为参考。
   - `tanh + a sin(wx + phi)` 的 amplitude/frequency/phase ablation。
   - `tanh_cos` 和 `tanh_rand` 作为 specificity/sanity control。
4. 机制分析：
   - FFN 前后 activation distribution；
   - tanh-like saturation rate；
   - layer-wise gradient norm/update norm；
   - error by horizon position；
   - solar day/night 分析；
   - volatility/turbulence/anomaly density 分桶误差。
5. 统计要求：
   - frozen aggregation；
   - paired tests 和 confidence intervals；
   - 多 activation / 多 factor 检验时，要有 multiple-comparison-aware interpretation；
   - Phase 1 后冻结 headline comparisons，避免事后挑结果。
6. 强负对照：
   - random/noise perturbation 不应系统性复制 sine 的收益；
   - output tanh 必须作为独立机制，不可和 hidden activation 混写。
7. 贡献形态：
   - 不是 “MSE 表格更好”，而是一个可复用的 factor-conditioned activation placement diagnostic protocol。

没有以下内容时，CCF-B 会比较弱：

- 现代模型泛化；
- 去重后的置信区间；
- 明确机制 probe；
- 展示 interaction effect 的图，而不是只有 global bars。

## 7. 风险与审稿攻击点

1. early signal 可能来自训练不稳定。
   - 应对：更多 seeds、paired deltas、相同训练预算、报告失败/异常。
2. 可能是 output range 或 normalization artifact。
   - 应对：强调默认 linear output；加入 output tanh 反事实。
3. Solar 和 ETT 机制不一致。
   - 应对：把这写成结果，即 dataset/horizon-conditioned behavior。
4. Lee 慢且不稳定。
   - 应对：Lee 作为 mechanism probe，不作为默认主方法，除非更多数据支持。
5. factor correlation 不是因果。
   - 应对：使用 “associated with”“conditional diagnostic”“interaction signal” 等措辞。
6. 多因子/多激活检验容易出现假阳性。
   - 应对：主文只预设 volatility 和 turbulence；其他因子放 appendix。
7. Informer-only 可能显得过时。
   - 应对：CCF-C 解释为 controlled substrate；CCF-B 补 PatchTST/iTransformer 或 DLinear。
8. “所有 bin 都赢 GELU” 没解释力。
   - 应对：重点看相对 tanh 的 high-minus-low improvement，而不是每个 bin 对 GELU 胜率。

## 8. 下一步实验计划

### 8.1 先做远程状态与结果冻结

开新任务前：

1. 检查远程服务器状态：
   - 可用候选：`10.21.53.62`、`10.21.53.113`、`10.21.53.142`、`10.21.53.162`、`10.21.53.82`；
   - 只在必要时重查：`10.21.53.156`；
   - 不要使用：`10.20.12.248`、`10.20.12.247`。
2. 拉回当前 `results/`、`logs/`、configs。
3. 生成统一 summary。
4. 冻结去重策略后再写任何 aggregate claim。

### 8.2 layer/output counterfactual 矩阵

核心 configs：

- hidden GELU，output linear。
- hidden tanh，output linear。
- encoder tanh，decoder GELU，output linear。
- encoder GELU，decoder tanh，output linear。
- hidden tanh，output tanh。
- hidden `tanh_sin a=0.01`，output linear。

优先 cells：

- Solar1: 24, 96。
- Solar5: 24, 96。
- ETTh1: 24, 168, 336。
- ETTh2: 24, 168, 336。

解释规则：

- 如果 output tanh 只在 ETT 有帮助、在 Solar 有害，这是 dataset/output-range mechanism，不是 hidden activation mechanism。
- 如果 encoder-only tanh 捕获大部分收益，论文可以把机制定位到 encoder FFN。
- 如果 decoder-only tanh 弱，不要把 encoder/decoder 平均成一句 “activation works”。

### 8.3 factor-conditioned perturbation 矩阵

主假设：

> 在更长 horizon 且高 volatility/high turbulence 输入窗口中，围绕 tanh 的轻微振荡扰动更可能降低误差。

必须计算：

- 每个样本/窗口：
  `delta_mse = mse_perturb - mse_tanh`。
- 每个因子：
  使用 input-window factor 分 low/mid/high bins。
- 报告：
  - 每个 bin 的平均 improvement；
  - high-minus-low improvement gradient；
  - paired confidence interval；
  - `activation:high_factor` interaction coefficient。

主因子：

- volatility。
- turbulence_score。

附录因子：

- anomaly_density_intensity。
- trend。
- mean_abs_change。
- range。
- skewness/kurtosis 只在稳定时使用。

### 8.4 oscillation specificity

矩阵保持克制：

- `tanh`。
- `tanh_sin a=0.005`。
- `tanh_sin a=0.01`。
- `tanh_sin a=0.05`。
- `tanh_cos a=0.01` 作为 supplement。
- `tanh_rand a=0.01` 作为 sanity control。
- Lee1/Lee3 小规模跑。

不要过度扩展 random/noise controls。只有当 sine 出现强正信号时，才值得扩大 sanity controls。

## 9. Related Work 写作建议

建议组织成四段：

1. Activation functions and signal propagation  
   写 ReLU/GELU/Swish/SELU/Deep Information Propagation，说明 activation 会影响 trainability 和 representation。
2. Placement and learnable activations  
   写 pre-activation ResNet、Transformer learnable activation、DiTAC/APA，说明 activation 的位置和可适应性重要。
3. Periodic activations  
   写 SIREN、Fourier features、Snake，说明 periodic inductive bias 已有基础，但场景不同。
4. LTSF context  
   写 Informer、Autoformer、FEDformer、DLinear、PatchTST、iTransformer，说明 LTSF 主要研究结构/分解/tokenization，而不是 activation placement + data-state factors。

可用句子：

> 不同于面向坐标表示或周期函数外推的 periodic activation 工作，我们将轻微振荡扰动作为 LTSF Transformer FFN 中的受控干预，并用输入窗口因子评估其条件有效性。

关于 tanh 与高波动的可用句子：

> 现有高波动金融时间序列研究并未形成 tanh 系统占优的共识；这使得我们对 tanh/bounded activation 的 factor-conditioned 分析更像一个待验证机制，而不是对已知经验规律的简单复现。

## 10. 来源清单

激活函数与信号传播：

- Nair and Hinton, "Rectified Linear Units Improve Restricted Boltzmann Machines", ICML 2010: https://icml.cc/Conferences/2010/papers/432.pdf
- Hendrycks and Gimpel, "Gaussian Error Linear Units (GELUs)", arXiv:1606.08415: https://arxiv.org/abs/1606.08415
- Ramachandran, Zoph, Le, "Searching for Activation Functions", arXiv:1710.05941: https://arxiv.org/abs/1710.05941
- Klambauer et al., "Self-Normalizing Neural Networks", NeurIPS 2017: https://proceedings.neurips.cc/paper/2017/hash/5d44ee6f2c3f71b73125876103c8f6c4-Abstract.html
- Schoenholz et al., "Deep Information Propagation", ICLR 2017: https://research.google/pubs/deep-information-propagation/
- He et al., "Delving Deep into Rectifiers", ICCV 2015: https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html
- He et al., "Identity Mappings in Deep Residual Networks", ECCV 2016 / arXiv:1603.05027: https://arxiv.org/abs/1603.05027
- LeCun et al., "Efficient BackProp", 1998: https://bottou.org/papers/lecun-98x
- Goodfellow, Bengio, Courville, Deep Learning, 2016: https://www.deeplearningbook.org/
- Chelly et al., "Trainable Highly-expressive Activation Functions", ECCV 2024 / arXiv:2407.07564: https://arxiv.org/abs/2407.07564
- Alexandridis et al., "Adaptive Parametric Activation", arXiv:2407.08567: https://arxiv.org/abs/2407.08567

周期/振荡激活：

- Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions", NeurIPS 2020: https://www.computationalimaging.org/publications/siren/
- Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains", NeurIPS 2020: https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html
- Ziyin, Hartwig, Ueda, "Neural Networks Fail to Learn Periodic Functions and How to Fix It", NeurIPS 2020 / arXiv:2006.08195: https://proceedings.neurips.cc/paper/2020/hash/1160453108d3e537255e9f7b931f4e90-Abstract.html
- Benbarka et al., "Seeing Implicit Neural Representations as Fourier Series", WACV 2022 / arXiv:2109.00249: https://arxiv.org/abs/2109.00249

Transformer FFN activation：

- Vaswani et al., "Attention Is All You Need", NeurIPS 2017: https://arxiv.org/abs/1706.03762
- Dauphin et al., "Language Modeling with Gated Convolutional Networks", ICML 2017: https://proceedings.mlr.press/v70/dauphin17a.html
- Shazeer, "GLU Variants Improve Transformer", arXiv:2002.05202: https://arxiv.org/abs/2002.05202
- Fang et al., "Transformers with Learnable Activation Functions", Findings of EACL 2023 / arXiv:2208.14111: https://aclanthology.org/2023.findings-eacl.181/

长序列时间序列预测：

- Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting", AAAI 2021: https://ojs.aaai.org/index.php/AAAI/article/view/17325
- Wu et al., "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting", NeurIPS 2021: https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html
- Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting", ICML 2022: https://icml.cc/virtual/2022/spotlight/17986
- Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023 / arXiv:2205.13504: https://arxiv.org/abs/2205.13504
- Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers", ICLR 2023: https://openreview.net/forum?id=Jbdc0vTOcol
- Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting", ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/hash/2ea18fdc667e0ef2ad82b2b4d65147ad-Abstract-Conference.html

高波动/金融时间序列与 activation：

- Kayim and Yilmaz, "Time Series Forecasting With Volatility Activation Function", IEEE Access 2022: https://doaj.org/article/d77a3ae36a7e4e949f2246e9c2ad7578
- Gomes, Ludermir, Lima, "Comparison of new activation functions in neural network for forecasting financial time series", Neural Computing and Applications 2011: https://www.cin.ufpe.br/~tbl/artigos/NeuralComput%26Applic-gecy-Final.pdf
- Nugraha, Budiman, Akbar, "Activation Function Sensitivity in LSTM-Based Peak Stock Price Forecasting for High-Volatility Financial Time Series", SISINFO 2026: https://jurnalunibi.unibi.ac.id/ojs/index.php/SisInfo/article/view/1492
- "Encoder-Decoder Based LSTM and GRU Architectures for Stocks and Cryptocurrency Prediction", Journal of Risk and Financial Management 2024: https://www.mdpi.com/1911-8074/17/5/200
- "Comparing the effectiveness of deep feedforward neural networks and shallow architectures for predicting stock price indices", Expert Systems with Applications 2020: https://www.sciencedirect.com/science/article/pii/S0957417419305305
- Chen et al., "TimeCatcher: A Variational Framework for Volatility-Aware Forecasting of Non-Stationary Time Series", arXiv:2601.20448: https://arxiv.org/abs/2601.20448

