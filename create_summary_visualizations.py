import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 设置样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 路径配置
analysis_path = Path("Daily_Factors_Analysis")
output_path = analysis_path

# 读取所有数据集
datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']
all_data = {}

for dataset in datasets:
    csv_file = analysis_path / f"{dataset}_daily_factors.csv"
    all_data[dataset] = pd.read_csv(csv_file)

print("创建汇总可视化...")

# ============================================================================
# 1. 关键因子对比雷达图
# ============================================================================
print("1. 生成关键因子雷达图...")

fig, axes = plt.subplots(2, 2, figsize=(16, 16), subplot_kw=dict(projection='polar'))
fig.suptitle('ETT Datasets - Key Factors Radar Chart', fontsize=18, fontweight='bold', y=0.98)

factors = ['mean', 'std', 'cv', 'trend', 'volatility', 'range']
colors = {'ETTh1': '#FF6B6B', 'ETTh2': '#4ECDC4', 'ETTm1': '#45B7D1', 'ETTm2': '#FFA07A'}

for idx, dataset in enumerate(datasets):
    ax = axes[idx // 2, idx % 2]

    # 获取因子值并归一化到0-1
    values = []
    for factor in factors:
        val = all_data[dataset][factor].mean()
        # 归一化（使用所有数据集的最大值）
        max_val = max([all_data[ds][factor].mean() for ds in datasets])
        min_val = min([all_data[ds][factor].mean() for ds in datasets])
        if max_val != min_val:
            normalized = (val - min_val) / (max_val - min_val)
        else:
            normalized = 0.5
        values.append(normalized)

    # 闭合雷达图
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(factors), endpoint=False).tolist()
    angles += angles[:1]

    # 绘制
    ax.plot(angles, values, 'o-', linewidth=2, color=colors[dataset], label=dataset)
    ax.fill(angles, values, alpha=0.25, color=colors[dataset])

    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f.upper() for f in factors], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title(f'{dataset}', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig(output_path / "summary_radar_chart.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 2. 数据集特征对比条形图
# ============================================================================
print("2. 生成特征对比条形图...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('ETT Datasets - Factor Comparison', fontsize=16, fontweight='bold')

factors_to_plot = ['mean', 'std', 'cv', 'trend', 'volatility', 'range']

for idx, factor in enumerate(factors_to_plot):
    ax = axes[idx // 3, idx % 3]

    # 收集数据
    data_to_plot = []
    labels = []
    colors_list = []

    for dataset in datasets:
        val = all_data[dataset][factor].mean()
        data_to_plot.append(val)
        labels.append(dataset)
        colors_list.append(colors[dataset])

    # 绘制条形图
    bars = ax.bar(labels, data_to_plot, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel(factor.upper(), fontsize=11, fontweight='bold')
    ax.set_title(f'{factor.upper()} Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 旋转x轴标签
    ax.set_xticklabels(labels, rotation=0, ha='center')

plt.tight_layout()
plt.savefig(output_path / "summary_bar_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. 时间序列对比（选择关键因子）
# ============================================================================
print("3. 生成时间序列对比图...")

fig, axes = plt.subplots(3, 1, figsize=(18, 12))
fig.suptitle('ETT Datasets - Time Series Comparison (First 100 Days)', fontsize=16, fontweight='bold')

factors_to_plot = ['mean', 'volatility', 'trend']

for idx, factor in enumerate(factors_to_plot):
    ax = axes[idx]

    for dataset in datasets:
        # 只绘制前100天
        days = all_data[dataset]['day'][:100]
        values = all_data[dataset][factor][:100]
        ax.plot(days, values, linewidth=2, alpha=0.8, label=dataset, color=colors[dataset])

    ax.set_xlabel('Day', fontsize=11)
    ax.set_ylabel(factor.upper(), fontsize=11, fontweight='bold')
    ax.set_title(f'{factor.upper()} over Time', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_path / "summary_timeseries_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 4. 分布对比小提琴图
# ============================================================================
print("4. 生成分布对比小提琴图...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('ETT Datasets - Factor Distribution Comparison', fontsize=16, fontweight='bold')

factors_to_plot = ['mean', 'std', 'cv', 'trend', 'volatility', 'range']

for idx, factor in enumerate(factors_to_plot):
    ax = axes[idx // 3, idx % 3]

    # 准备数据
    data_list = []
    labels = []

    for dataset in datasets:
        data_list.append(all_data[dataset][factor].values)
        labels.append(dataset)

    # 绘制小提琴图
    parts = ax.violinplot(data_list, positions=range(len(datasets)),
                          showmeans=True, showmedians=True)

    # 设置颜色
    for pc, dataset in zip(parts['bodies'], datasets):
        pc.set_facecolor(colors[dataset])
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel(factor.upper(), fontsize=11, fontweight='bold')
    ax.set_title(f'{factor.upper()} Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_path / "summary_violin_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 5. 相关性矩阵对比
# ============================================================================
print("5. 生成相关性矩阵对比图...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('ETT Datasets - Factor Correlation Matrices', fontsize=16, fontweight='bold')

numeric_factors = ['mean', 'std', 'cv', 'trend', 'volatility', 'range']

for idx, dataset in enumerate(datasets):
    ax = axes[idx // 2, idx % 2]

    corr_matrix = all_data[dataset][numeric_factors].corr()

    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                vmin=-1, vmax=1)

    ax.set_title(f'{dataset} - Correlation Matrix', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(output_path / "summary_correlation_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 6. 统计摘要表格图
# ============================================================================
print("6. 生成统计摘要表格...")

fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('tight')
ax.axis('off')

# 创建表格数据
table_data = []
table_data.append(['Dataset', 'Mean Load', 'Std Dev', 'CV', 'Trend', 'Volatility', 'Range'])

for dataset in datasets:
    row = [
        dataset,
        f"{all_data[dataset]['mean'].mean():.2f}",
        f"{all_data[dataset]['std'].mean():.2f}",
        f"{all_data[dataset]['cv'].mean():.3f}",
        f"{all_data[dataset]['trend'].mean():.4f}",
        f"{all_data[dataset]['volatility'].mean():.4f}",
        f"{all_data[dataset]['range'].mean():.2f}"
    ]
    table_data.append(row)

# 添加对比行
table_data.append(['', '', '', '', '', '', ''])
table_data.append(['Comparison', '', '', '', '', '', ''])
table_data.append(['h vs m (volatility)',
                  '', '', '', '',
                  f"h: {(all_data['ETTh1']['volatility'].mean() + all_data['ETTh2']['volatility'].mean())/2:.3f}",
                  ''])
table_data.append(['',
                  '', '', '', '',
                  f"m: {(all_data['ETTm1']['volatility'].mean() + all_data['ETTm2']['volatility'].mean())/2:.3f}",
                  ''])

# 创建表格
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# 设置表头样式
for i in range(7):
    cell = table[(0, i)]
    cell.set_facecolor('#4ECDC4')
    cell.set_text_props(weight='bold', color='white')

# 设置数据集行颜色
for i, dataset in enumerate(datasets, 1):
    cell = table[(i, 0)]
    cell.set_facecolor(colors[dataset])
    cell.set_text_props(weight='bold', color='white')

plt.title('ETT Datasets - Statistical Summary', fontsize=16, fontweight='bold', pad=20)
plt.savefig(output_path / "summary_statistics_table.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 7. h系列 vs m系列 对比
# ============================================================================
print("7. 生成h系列 vs m系列对比图...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Hourly (h) vs 15-min (m) Series Comparison', fontsize=16, fontweight='bold')

# 对比因子
comparison_factors = ['volatility', 'trend', 'mean_abs_change', 'range']

for idx, factor in enumerate(comparison_factors):
    ax = axes[idx // 2, idx % 2]

    # h系列数据
    h_data = [all_data['ETTh1'][factor].mean(), all_data['ETTh2'][factor].mean()]
    # m系列数据
    m_data = [all_data['ETTm1'][factor].mean(), all_data['ETTm2'][factor].mean()]

    x = np.arange(2)
    width = 0.35

    bars1 = ax.bar(x - width/2, h_data, width, label='h-series (24 points/day)',
                   color='#FF6B6B', alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, m_data, width, label='m-series (96 points/day)',
                   color='#45B7D1', alpha=0.7, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel(factor.upper(), fontsize=11, fontweight='bold')
    ax.set_title(f'{factor.upper()} Comparison', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Dataset 1', 'Dataset 2'])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_path / "summary_h_vs_m_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================================
# 8. 季节性对比
# ============================================================================
print("8. 生成季节性对比图...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Seasonal Analysis - Mean Load by Season', fontsize=16, fontweight='bold')

for idx, dataset in enumerate(datasets):
    ax = axes[idx // 2, idx % 2]

    df = all_data[dataset].copy()
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })

    # 按季节统计
    seasonal_mean = df.groupby('season')['mean'].mean()
    seasonal_std = df.groupby('season')['mean'].std()

    seasons = ['Spring', 'Summer', 'Autumn', 'Winter']
    means = [seasonal_mean.get(s, 0) for s in seasons]
    stds = [seasonal_std.get(s, 0) for s in seasons]

    bars = ax.bar(seasons, means, yerr=stds, capsize=5,
                  color=colors[dataset], alpha=0.7, edgecolor='black', linewidth=1.5)

    # 添加数值标签
    for bar, mean_val in zip(bars, means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean_val:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Mean Load', fontsize=11, fontweight='bold')
    ax.set_title(f'{dataset} - Seasonal Mean Load', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_path / "summary_seasonal_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("所有汇总可视化已生成!")
print("="*80)
print("\n生成的文件:")
print("  1. summary_radar_chart.png - 关键因子雷达图")
print("  2. summary_bar_comparison.png - 因子对比条形图")
print("  3. summary_timeseries_comparison.png - 时间序列对比")
print("  4. summary_violin_comparison.png - 分布对比小提琴图")
print("  5. summary_correlation_comparison.png - 相关性矩阵对比")
print("  6. summary_statistics_table.png - 统计摘要表格")
print("  7. summary_h_vs_m_comparison.png - h系列vs m系列对比")
print("  8. summary_seasonal_comparison.png - 季节性对比")
print("\n所有文件保存在: Daily_Factors_Analysis/")
print("="*80)
