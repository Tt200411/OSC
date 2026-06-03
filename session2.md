你正在接手一个已有研究项目。请不要从零开始猜测，先按下面信息恢复上下文，然后继续推进。

  项目目录：
  /Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main

  当前研究主线：
  我们研究 Informer/长序列时间序列预测中，激活函数扰动是否能提升结果，以及“什么时候有效”。核心问题不是单纯刷分，而是解释：
  1. 轻微振荡扰动，如 a*sin(x)、Lee1/Lee3，是否真的有效？
  2. 它是否只在长序列、高 volatility / turbulence 区间有效？
  3. bounded activation、oscillatory activation、output activation 的作用机制是否不同？
  4. 这些现象是否能支持 CCF-B / CCF-C 级别论文。

  重要约束：
  - 本地只做代码整理、分析、写作、结果汇总，不在本地跑训练。
  - 训练必须远程跑 GPU 服务器。
  - 不要使用 10.20.12.248 开新任务，用户明确说别用了。
  - 10.20.12.247 暂时也不要用。
  - 不要把 SSH 密码写进代码、脚本、日志或结果文件。
  - 结果必须诚实，不要编造不存在的实验。
  - 如果实验只是 early signal，要明确标为 early signal，不能说已经统计证明。

  SSH 信息：
  - 用户名：testsv
  - 密码：<redacted; transient only>
  - 远程工作目录：~/project/osc_informer
  - 服务器：
    - 10.21.53.62：上次已可用，跑过 layer probe
    - 10.21.53.113：上次已可用，跑过 ETTh2:168 layer probe
    - 10.21.53.142：上次已可用，跑过 Solar1:24 layer probe
    - 10.21.53.162：上次已可用，跑过 Solar5:24 layer probe
    - 10.21.53.82：上次已可用，跑过 ETTh1:168 layer probe
    - 10.21.53.156：上次 SSH timeout，需要重新验证
    - 10.20.12.248：不要开新任务
    - 10.20.12.247：不要用
  - 上服务器前先检查 nvidia-smi、磁盘、已有 python/torch/cuda 环境和正在运行的任务。
  - 如需自动登录，可以临时用 expect/sshpass，但不能把密码落盘。

  必须先读的本地文件：
  1. NARRATIVE_REPORT.md
  2. PAPER_PLAN.md
  3. lee_ocil/models/model.py
  4. lee_ocil/scripts/run_phase1_layer_activation_probe.sh
  5. lee_ocil/scripts/run_phase1_tanh_specificity_probe.sh
  6. lee_ocil/scripts/run_phase1_oscillation_forms_probe.sh
  7. lee_ocil/scripts/run_phase1_mechanism_probe.sh
  8. lee_ocil/scripts/phase1_heartbeat_monitor.sh
  9. scripts/sync_phase1_remote.sh
  10. scripts/fetch_phase1_results.sh
  11. paper/main.tex
  12. paper/sections/*.tex
  13. paper/figures/*.tex
  14. paper/PAPER_CLAIM_AUDIT.json
  15. paper/CITATION_AUDIT.json
  16. paper/KILL_ARGUMENT.json
  17. paper/PAPER_WRITING_PIPELINE_REPORT.md
  18. .aris/ 下如果存在，也要读 manifest / traces / audit 相关文件

  已经完成的重要代码/论文工作：
  - 已经把 Informer 的 activation 从硬编码 lee 改成可配置。
  - 当前 projection/output head 默认是线性：
    lee_ocil/models/model.py 里 self.projection = nn.Linear(...)
    也就是说此前结果不是因为输出层默认 tanh 导致的。
  - 已新增/已有控制项：
    --encoder_activation
    --decoder_activation
    --output_activation，默认 linear，可选 tanh
    activation_signature 写入结果
  - 已新增/已有脚本：
    lee_ocil/scripts/run_phase1_layer_activation_probe.sh
  - 已修复 sync 脚本，避免删除远程 .venv*：
    scripts/sync_phase1_remote.sh
  - 之前已 push 的 commit：
    848b6b1 Add layer activation mechanism probe
    348ec83 Avoid deleting remote venv variants during sync

  上次观察到的 early layer-probe 信号：
  Solar1 pred_len=96 seed2024：
  - GELU hidden + linear output: MSE 0.281079
  - tanh hidden + linear output: 0.259251
  - encoder tanh / decoder GELU / linear output: 0.241045，最好
  - encoder GELU / decoder tanh / linear output: 0.275032
  - tanh hidden + tanh output: 0.409333，明显更差
  - tanh_sin a=0.01 hidden + linear output: 0.259195
  解释：Solar 上 output tanh 有害，encoder-side bounded hidden activation 可能是关键。

  ETTh2 pred_len=168，10.21.53.113，seeds 2024-2026：
  - seed2024:
    GELU 11.006948
    tanh hidden linear 4.358913
    encoder-only tanh 6.112610
    decoder-only tanh 10.889180
    tanh hidden output-tanh 3.605213
    tanh_sin 4.399738
  - seed2025:
    GELU 6.642311
    tanh hidden linear 3.213023
    encoder-only tanh 4.340827
    decoder-only tanh 6.528734
    output-tanh 2.956166
    tanh_sin 3.435526
  - seed2026:
    GELU 6.449427
    tanh hidden linear 4.362999
    encoder-only tanh 4.277670
    decoder-only tanh 6.834924
    output-tanh 2.773134
  解释：ETT 上 output tanh 可能有额外帮助，但这是 dataset-specific 机制；不能混同于此前 hidden activation 的提升。

  ETTh1 pred_len=168，10.21.53.82：
  - seed2024:
    GELU 1.261057
    tanh hidden linear 1.171317
    encoder-only tanh 1.182373
    decoder-only tanh 1.244209
    output-tanh 1.011842
    tanh_sin 1.139032
  - seed2025:
    GELU 1.371655
    tanh hidden linear 1.298101
    encoder-only tanh 1.261887
    decoder-only tanh 1.453466
    output-tanh 1.172568
  解释：ETT 上 output tanh 可能帮忙，但此前模型默认不是 output tanh。

  Solar1 pred_len=24 early：
  - GELU 0.174010
  - tanh hidden linear 0.165131
  - encoder-only tanh 0.164675

  Solar5 pred_len=24 early：
  - GELU 0.127426
  - tanh hidden linear 0.110828

  当前最重要的科学结论草案：
  - 不能简单说“Lee/扰动永远有效”。
  - 更合理的 claim 是：
    activation placement 是时间序列 Transformer 中被低估的机制变量；
    bounded hidden activations 和 output bounding 是不同机制；
    轻微振荡扰动可能只在高 volatility / turbulence、长 horizon 或特定数据状态下有收益；
    需要用因子分桶和 interaction 分析证明“高波动区收益更高”，而不是只看每个桶相对 GELU 的胜率。
  - 如果所有周期位置/所有因子桶都赢，那么因子分桶实验没有解释力；要重点看 high-minus-low gradient、bin interaction、改善幅度随
  volatility/turbulence 的单调性或差异。

  当前用户最新任务：
  用户要求搜索已有论文，来源包括 arXiv 和 CCF 会议，看看是否有人讨论过：
  1. 激活函数对神经网络的深层意义
  2. 激活函数在不同位置/layer/output head 上产生不同结果
  3. bounded/oscillatory activation，如 tanh、sin、Snake、SIREN
  4. Transformer / time-series 里 FFN activation 的作用
  5. 用这些文献帮助构思：基于我们 early signal，还需要补充什么实验，才能组成一篇足够支持 CCF-B 或 CCF-C 投稿的稿件

  用户还要求使用已安装的 auto research 相关 skill。注意：
  - 本地没有精确叫 “Auto-claude-code-research-in-sleep” 的 skill。
  - 应该说明这个 skill 不存在，然后使用可用 fallback：
    research-lit
    arxiv
    semantic-scholar
    novelty-check
    result-to-claim
    experiment-plan / ablation-planner
    paper-writing / paper-compile / citation-audit / paper-claim-audit

  可用 skill 路径：
  - /Users/tangbao/.codex/skills/research-lit/SKILL.md
  - /Users/tangbao/.codex/skills/arxiv/SKILL.md
  - /Users/tangbao/.codex/skills/semantic-scholar/SKILL.md
  - /Users/tangbao/.codex/skills/novelty-check/SKILL.md
  - /Users/tangbao/.codex/skills/result-to-claim/SKILL.md
  - /Users/tangbao/.codex/skills/ablation-planner/SKILL.md
  - /Users/tangbao/.codex/skills/analyze-results/SKILL.md
  - /Users/tangbao/.codex/skills/monitor-experiment/SKILL.md
  - /Users/tangbao/.codex/skills/paper-writing/SKILL.md

  文献检索方向，需要优先核验：
  - Hendrycks & Gimpel, Gaussian Error Linear Units, arXiv:1606.08415
  - Ramachandran et al., Searching for Activation Functions, arXiv:1710.05941
  - Klambauer et al., Self-Normalizing Neural Networks, NeurIPS 2017
  - Schoenholz / Poole 等，Deep Information Propagation, ICLR 2017 相关
  - He et al., Identity Mappings in Deep Residual Networks, ECCV 2016，activation placement / pre-activation
  - Sitzmann et al., SIREN, NeurIPS 2020，periodic activations
  - “Neural Networks Fail to Learn Periodic Functions and How to Fix It”，Snake activation，需核验作者和 venue
  - Tancik et al., Fourier Features Let Networks Learn High Frequency Functions, NeurIPS 2020
  - Shazeer, GLU Variants Improve Transformer, arXiv 2020
  - Transformers with Learnable Activation Functions / rational activations，需核验
  - Informer, AAAI 2021
  - Autoformer, NeurIPS 2021
  - FEDformer, ICML 2022
  - PatchTST, ICLR 2023
  - DLinear / Are Transformers Effective for Time Series Forecasting，需核验 venue
  - iTransformer, ICLR 2024
  - Deep Learning Book / Efficient BackProp 对 regression output linear units 的论述

  请输出或落盘一个 literature/memo：
  建议保存到：
  .aris/literature_activation_position_2026-05-16.md
  内容包括：
  1. 已有工作分桶：
     - activation as optimization / representation / signal propagation
     - activation placement / pre-activation / output units
     - periodic / oscillatory activations
     - Transformer FFN activations
     - long-horizon time-series forecasting context
  2. 和我们工作的差异：
     - 目前看没有直接做 “factor-conditioned activation placement and oscillatory perturbation for Informer/LTSF” 的完全相同工作
     - 我们的新颖点应收窄为 activation placement + factor-conditioned effectiveness，而不是泛泛地发明新 activation
  3. 支撑 CCF-C 需要补什么：
     - ETTh1/ETTh2/Solar1/Solar5
     - 每个至少 2-3 个 horizon
     - 至少 3 seeds
     - GELU/ReLU/tanh/tanh_sin/Lee1 or Lee3/encoder-only/decoder-only/output-linear-vs-tanh
     - 因子分桶和 high-minus-low gradient
     - paired statistical tests / confidence intervals
  4. 支撑 CCF-B 需要补什么：
     - 更多 dataset，例如 ETTm1/ETTm2/ECL/Weather
     - 至少一个现代模型泛化，如 PatchTST、DLinear、iTransformer 中可行者
     - 系统性 ablation：amplitude, frequency, phase, activation placement, output activation
     - 机制分析：activation distribution、gradient norm、saturation rate、prediction error by horizon/day-night/high-volatility
     - 严格统计检验和消融
  5. 风险：
     - early signal 可能来自训练不稳定或 normalization/output range
     - Solar 和 ETT 机制不一致
     - Lee 太慢，不适合大规模主实验，可以只做小规模 mechanism probe
     - 不能把 factor correlation 说成因果

  下一步实验建议：
  6. 先检查所有远程当前任务状态，拉回已有 results/logs/config。
  7. 生成统一 summary，去重，按 dataset/horizon/seed/config 聚合。
  8. 完成 layer/output activation counterfactual：
     - hidden=tanh/output=linear
     - hidden=tanh/output=tanh
     - encoder=tanh decoder=gelu output=linear
     - encoder=gelu decoder=tanh output=linear
  9. 针对用户核心猜想做专门实验：
     “长序列高 volatility 时间段上，a=0.01 轻微扰动更有帮助。”
     不能只看每个桶胜率，要计算：
     - 每个样本或窗口的 delta MSE = MSE_perturb - MSE_baseline
     - 按 input-window volatility/turbulence 分桶
     - high-bin improvement - low-bin improvement
     - interaction regression:
       improvement ~ activation + factor_bin + activation:factor_bin + dataset + horizon
  10. 对 tanh 基础上的 a*sin(x) 和 Lee1/Lee3 做特异性补实验：
     - tanh
     - tanh_sin a=0.005/0.01/0.05
     - tanh_cos a=0.01 可作为 supplement
     - random/noise perturbation 只作 sanity control，不要扩大太多
     - Lee1/Lee3 小规模跑，证明其是否区别于 tanh + sin
  11. 写论文时主 claim 要克制：
     - “Activation placement and mild oscillatory perturbation can improve selected LTSF regimes”
     - “effectiveness is factor-dependent, especially in high volatility/turbulence long-horizon windows”
     - 不要说 universal improvement。

  如果要继续写作：
  - 当前 paper/ 已经存在 main.tex 和 sections。
  - 之前 paper audit 状态不是完全通过：
    - paper/PAPER_CLAIM_AUDIT.json：WARN
    - paper/CITATION_AUDIT.json：WARN
    - paper/KILL_ARGUMENT.json：FAIL
  - 所以不能说 submission-ready。
  - 应先补 literature memo 和 result-to-claim，再修 paper sections，再 compile。

  工作方式：
  - 先给简短状态更新。
  - 需要查最新论文必须联网检索，优先官方/arXiv/Semantic Scholar/会议页面。
  - 远程训练只用空闲服务器，不要压垮已有进程。
  - 完成阶段性工作后 git status，必要时 commit/push。

  这份 prompt 已经把当前记忆、约束、服务器、文件、已知 early signal 和下一步任务都压进去了。新进程拿到后应先读文件和复核服务器状态，
  再继续文献检索与实验设计。
