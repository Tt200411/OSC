# ETT数据集每日因子分析报告

## 📊 项目概述

本项目对ETT-small数据集（ETTh1, ETTh2, ETTm1, ETTm2）的OT列（目标列）进行了按日切分的统计因子计算和可视化分析。

### 数据切分方式
- **ETTh系列** (ETTh1, ETTh2): 每天24个数据点（小时级数据）
- **ETTm系列** (ETTm1, ETTm2): 每天96个数据点（15分钟级数据）

### 分析周期
- 所有数据集: 725天（2016-07-01 至 2018-06-25）

---

## 📁 文件结构

```
Daily_Factors_Analysis/
├── README.md                              # 本文档
├── FACTORS_EXPLANATION.md                 # 因子详细说明文档
│
├── ETTh1_daily_factors.csv                # ETTh1每日因子数据
├── ETTh2_daily_factors.csv                # ETTh2每日因子数据
├── ETTm1_daily_factors.csv                # ETTm1每日因子数据
├── ETTm2_daily_factors.csv                # ETTm2每日因子数据
├── all_datasets_summary.csv               # 所有数据集汇总统计
│
├── ETTh1_factors_timeseries.png           # ETTh1因子时间序列
├── ETTh1_factors_distribution.png         # ETTh1因子分布
├── ETTh1_factors_correlation.png          # ETTh1因子相关性
├── ETTh1_factors_boxplot.png              # ETTh1因子箱线图
│
├── ETTh2_factors_timeseries.png           # ETTh2因子时间序列
├── ETTh2_factors_distribution.png         # ETTh2因子分布
├── ETTh2_factors_correlation.png          # ETTh2因子相关性
├── ETTh2_factors_boxplot.png              # ETTh2因子箱线图
│
├── ETTm1_factors_timeseries.png           # ETTm1因子时间序列
├── ETTm1_factors_distribution.png         # ETTm1因子分布
├── ETTm1_factors_correlation.png          # ETTm1因子相关性
├── ETTm1_factors_boxplot.png              # ETTm1因子箱线图
│
├── ETTm2_factors_timeseries.png           # ETTm2因子时间序列
├── ETTm2_factors_distribution.png         # ETTm2因子分布
├── ETTm2_factors_correlation.png          # ETTm2因子相关性
├── ETTm2_factors_boxplot.png              # ETTm2因子箱线图
│
├── all_datasets_comparison.png            # 所有数据集对比
├── summary_radar_chart.png                # 关键因子雷达图
├── summary_bar_comparison.png             # 因子对比条形图
├── summary_timeseries_comparison.png      # 时间序列对比
├── summary_violin_comparison.png          # 分布对比小提琴图
├── summary_correlation_comparison.png     # 相关性矩阵对比
├── summary_statistics_table.png           # 统计摘要表格
├── summary_h_vs_m_comparison.png          # h系列vs m系列对比
└── summary_seasonal_comparison.png        # 季节性对比
```

---

## 🔍 计算的因子

### 1. 基础统计量
- **mean**: 每日平均值
- **std**: 标准差
- **median**: 中位数
- **min/max**: 最小值/最大值
- **range**: 极差 (max - min)

### 2. 变异性指标
- **cv**: 变异系数 (std / mean)
- **iqr**: 四分位距 (Q3 - Q1)

### 3. 分布特征
- **skewness**: 偏度（分布对称性）
- **kurtosis**: 峰度（分布尖峭程度）

### 4. 趋势指标
- **trend**: 线性趋势斜率
- **first_last_diff**: 首尾差值
- **first_last_ratio**: 首尾比率

### 5. 波动性指标
- **volatility**: 一阶差分的标准差
- **mean_abs_change**: 平均绝对变化
- **max_abs_change**: 最大绝对变化

### 6. 百分位数
- **p10, p25, p75, p90**: 第10、25、75、90百分位数

详细说明请参阅 [FACTORS_EXPLANATION.md](FACTORS_EXPLANATION.md)

---

## 📈 关键发现

### 1. 负荷水平对比
| 数据集 | 平均负荷 | 说明 |
|--------|----------|------|
| ETTh1  | 13.33    | 低负荷 |
| ETTh2  | 26.59    | 高负荷（约为ETTh1的2倍）|
| ETTm1  | 13.32    | 低负荷 |
| ETTm2  | 26.59    | 高负荷（约为ETTm1的2倍）|

**结论**: h系列和m系列的负荷水平一致，说明数据来源相同，只是采样频率不同。

### 2. 波动性对比
| 数据集 | 波动率 | 排名 |
|--------|--------|------|
| ETTm2  | 0.3368 | 最低 ⭐ |
| ETTm1  | 0.4035 | 低 |
| ETTh1  | 0.8204 | 高 |
| ETTh2  | 1.1612 | 最高 |

**关键洞察**:
- m系列（96点/天）波动率显著低于h系列（24点/天）
- 更细的时间粒度使相邻点变化更平滑
- **预测建议**: m系列更适合精细化预测，h系列更适合捕捉宏观趋势

### 3. 趋势特征
| 数据集 | 平均趋势 | 上升趋势日占比 |
|--------|----------|----------------|
| ETTh1  | 0.0649   | 71.7% |
| ETTh2  | 0.2477   | 83.6% ⭐ |
| ETTm1  | 0.0153   | 70.9% |
| ETTm2  | 0.0590   | 83.3% |

**结论**:
- ETTh2/ETTm2具有明显的上升趋势
- ETTh1/ETTm1相对平稳

### 4. 相对波动性（变异系数CV）
| 数据集 | CV值 | 稳定性 |
|--------|------|--------|
| ETTm2  | 0.1607 | 最稳定 ⭐ |
| ETTh2  | 0.1613 | 稳定 |
| ETTh1  | 0.2172 | 中等 |
| ETTm1  | 0.3111 | 波动较大 |

**结论**: ETTh2/ETTm2相对更稳定（CV较小）

### 5. 因子相关性
所有数据集中发现的高相关性因子对（|r| > 0.7）：
- **std ↔ range**: r ≈ 0.94-0.99（极强相关）
- **std ↔ iqr**: r ≈ 0.89-0.98（极强相关）
- **range ↔ iqr**: r ≈ 0.78-0.93（强相关）
- **volatility ↔ range**: r ≈ 0.76（强相关，仅部分数据集）

**应用建议**: 在特征工程中可以考虑去除冗余特征（如保留std，去除range和iqr）

### 6. 季节性特征（以ETTh1为例）
| 季节 | 平均负荷 | 平均波动率 |
|------|----------|------------|
| 夏季 | 22.04    | 0.94（最高）|
| 秋季 | 13.99    | 0.79 |
| 春季 | 11.26    | 0.83 |
| 冬季 | 6.11     | 0.72（最低）|

**结论**: 夏季负荷最高且波动最大，冬季负荷最低且波动最小

---

## 🎯 应用建议

### 1. 预测模型选择
- **高波动数据** (ETTh2, volatility=1.16):
  - 推荐使用鲁棒模型：LSTM、Transformer、GRU
  - 需要更多的历史数据

- **低波动数据** (ETTm2, volatility=0.34):
  - 简单模型即可：ARIMA、线性回归
  - 较少的历史数据即可获得好效果

### 2. 特征工程
- **趋势明显** (ETTh2, trend=0.25):
  - 添加趋势特征、时间特征
  - 考虑差分处理

- **趋势平稳** (ETTm1, trend=0.015):
  - 关注周期性特征
  - 季节性分解

### 3. 异常检测阈值
- 使用 `mean ± 3*std` 作为初始阈值
- 结合IQR方法: `[Q1-1.5*IQR, Q3+1.5*IQR]`
- 高波动数据集需要更宽松的阈值

### 4. 数据预处理
- **高CV数据**: 考虑归一化或标准化
- **偏态分布**: 考虑对数变换或Box-Cox变换
- **高相关性因子**: 进行特征选择，去除冗余

### 5. 时间粒度选择
- **实时控制**: 使用m系列（96点/天），响应更及时
- **长期规划**: 使用h系列（24点/天），计算效率更高
- **混合策略**: 短期预测用m系列，长期预测用h系列

---

## 💻 使用方法

### 1. 读取因子数据
```python
import pandas as pd

# 读取ETTh1每日因子
df = pd.read_csv('ETTh1_daily_factors.csv')

# 查看前5天
print(df.head())

# 查看统计摘要
print(df.describe())
```

### 2. 筛选特定条件的日期
```python
# 筛选高波动日（波动率 > 90分位数）
high_volatility_days = df[df['volatility'] > df['volatility'].quantile(0.9)]

# 筛选上升趋势日
uptrend_days = df[df['trend'] > 0]

# 筛选夏季数据
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
summer_days = df[df['month'].isin([6, 7, 8])]
```

### 3. 因子用于预测
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 使用因子预测下一天的平均负荷
features = df[['mean', 'std', 'cv', 'trend', 'volatility', 'range']]
target = df['mean'].shift(-1)  # 下一天的平均负荷

# 去除最后一行（没有下一天数据）
features = features[:-1]
target = target[:-1]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# 训练模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测
predictions = model.predict(X_test)

# 评估
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")
```

### 4. 交互式分析工具
```bash
# 在项目根目录运行

# 查看所有数据集摘要
python view_factor_analysis.py

# 查看数据集对比
python view_factor_analysis.py compare

# 查看Top 10高波动日
python view_factor_analysis.py top ETTh1 volatility 10

# 查看相关性分析
python view_factor_analysis.py corr ETTm1

# 查看季节性分析
python view_factor_analysis.py season ETTh2
```

---

## 📊 可视化说明

### 单数据集可视化（每个数据集4张图）

1. **factors_timeseries.png**: 关键因子随时间变化
   - 展示mean, std, cv, trend, volatility, range的时间序列
   - 用于识别长期趋势和周期性

2. **factors_distribution.png**: 因子分布特征
   - 直方图 + KDE曲线
   - 显示均值和中位数
   - 用于了解因子的统计分布

3. **factors_correlation.png**: 因子相关性热图
   - 展示所有因子之间的相关系数
   - 用于特征选择和冗余分析

4. **factors_boxplot.png**: 因子箱线图
   - 展示因子的离群值
   - 用于异常检测

### 汇总对比可视化（8张图）

1. **summary_radar_chart.png**: 关键因子雷达图
   - 4个数据集的多维对比
   - 直观展示各数据集特征

2. **summary_bar_comparison.png**: 因子对比条形图
   - 6个关键因子的数值对比
   - 清晰展示差异

3. **summary_timeseries_comparison.png**: 时间序列对比
   - 前100天的mean, volatility, trend对比
   - 展示动态变化趋势

4. **summary_violin_comparison.png**: 分布对比小提琴图
   - 展示因子的分布形状
   - 对比不同数据集的分布差异

5. **summary_correlation_comparison.png**: 相关性矩阵对比
   - 4个数据集的相关性矩阵并排展示
   - 识别共性和差异

6. **summary_statistics_table.png**: 统计摘要表格
   - 关键统计量的表格展示
   - 便于快速查阅

7. **summary_h_vs_m_comparison.png**: h系列vs m系列对比
   - 对比不同时间粒度的影响
   - 展示采样频率对因子的影响

8. **summary_seasonal_comparison.png**: 季节性对比
   - 4个数据集的季节性负荷对比
   - 识别季节性模式

---

## 🔧 技术细节

### 计算环境
- Python 3.x
- pandas, numpy, matplotlib, seaborn, scipy

### 计算脚本
- `calculate_daily_factors.py`: 主计算脚本
- `create_summary_visualizations.py`: 汇总可视化脚本
- `view_factor_analysis.py`: 交互式分析工具

### 数据质量
- 所有数据集均为725天完整数据
- 无缺失值
- 按日切分精确对齐

---

## 📚 参考文献

1. ETT数据集来源: [Informer论文](https://arxiv.org/abs/2012.07436)
2. 时间序列特征工程: [tsfresh文档](https://tsfresh.readthedocs.io/)
3. 统计分析方法: [scipy.stats文档](https://docs.scipy.org/doc/scipy/reference/stats.html)

---

## 📝 更新日志

- **2026-02-12**: 初始版本
  - 计算725天的每日因子
  - 生成17张单数据集可视化
  - 生成8张汇总对比可视化
  - 创建交互式分析工具
  - 编写详细文档

---

## 📧 联系方式

如有问题或建议，请联系项目维护者。

---

## 📄 许可证

本分析结果遵循原ETT数据集的许可证。

---

**生成时间**: 2026-02-12
**数据来源**: COTN/ETT-small
**分析工具**: Python 3.x + pandas + matplotlib + seaborn
