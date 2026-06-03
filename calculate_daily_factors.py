import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 路径配置
base_path = Path("/Users/tangbao/project/Oscillatory-Neural-Behavior-Leveraging-Dynamic-Activations-for-Improved-Learning-main")
data_path = base_path / "COTN/ETT-small"
output_path = base_path / "Daily_Factors_Analysis"
output_path.mkdir(exist_ok=True)

# 数据集配置
datasets_config = {
    'ETTh1': {'file': 'ETTh1.csv', 'points_per_day': 24, 'freq': 'hourly'},
    'ETTh2': {'file': 'ETTh2.csv', 'points_per_day': 24, 'freq': 'hourly'},
    'ETTm1': {'file': 'ETTm1.csv', 'points_per_day': 96, 'freq': '15min'},
    'ETTm2': {'file': 'ETTm2.csv', 'points_per_day': 96, 'freq': '15min'}
}

def calculate_daily_factors(ot_values):
    """
    计算单日OT数据的统计因子

    参数:
        ot_values: 单日的OT值数组

    返回:
        包含各种统计因子的字典
    """
    factors = {
        # 基础统计量
        'mean': np.mean(ot_values),
        'std': np.std(ot_values),
        'median': np.median(ot_values),
        'min': np.min(ot_values),
        'max': np.max(ot_values),
        'range': np.max(ot_values) - np.min(ot_values),

        # 变异性指标
        'cv': np.std(ot_values) / np.mean(ot_values) if np.mean(ot_values) != 0 else 0,  # 变异系数
        'iqr': np.percentile(ot_values, 75) - np.percentile(ot_values, 25),  # 四分位距

        # 分布特征
        'skewness': pd.Series(ot_values).skew(),  # 偏度
        'kurtosis': pd.Series(ot_values).kurtosis(),  # 峰度

        # 趋势指标
        'trend': np.polyfit(range(len(ot_values)), ot_values, 1)[0],  # 线性趋势斜率
        'first_last_diff': ot_values[-1] - ot_values[0],  # 首尾差值
        'first_last_ratio': ot_values[-1] / ot_values[0] if ot_values[0] != 0 else 0,  # 首尾比率

        # 波动性指标
        'volatility': np.std(np.diff(ot_values)),  # 一阶差分的标准差
        'mean_abs_change': np.mean(np.abs(np.diff(ot_values))),  # 平均绝对变化
        'max_abs_change': np.max(np.abs(np.diff(ot_values))),  # 最大绝对变化

        # 百分位数
        'p25': np.percentile(ot_values, 25),
        'p75': np.percentile(ot_values, 75),
        'p90': np.percentile(ot_values, 90),
        'p10': np.percentile(ot_values, 10),
    }

    return factors

def process_dataset(dataset_name, config):
    """处理单个数据集"""
    print(f"\n{'='*60}")
    print(f"处理数据集: {dataset_name}")
    print(f"{'='*60}")

    # 读取数据
    file_path = data_path / config['file']
    df = pd.read_csv(file_path)
    points_per_day = config['points_per_day']

    print(f"数据形状: {df.shape}")
    print(f"每天数据点数: {points_per_day}")

    # 提取OT列
    ot_data = df['OT'].values
    total_points = len(ot_data)
    num_days = total_points // points_per_day

    print(f"总数据点数: {total_points}")
    print(f"完整天数: {num_days}")

    # 按日切分并计算因子
    daily_factors_list = []

    for day_idx in range(num_days):
        start_idx = day_idx * points_per_day
        end_idx = start_idx + points_per_day
        daily_ot = ot_data[start_idx:end_idx]

        factors = calculate_daily_factors(daily_ot)
        factors['day'] = day_idx + 1
        factors['date'] = df.iloc[start_idx]['date']
        daily_factors_list.append(factors)

    # 转换为DataFrame
    factors_df = pd.DataFrame(daily_factors_list)

    # 保存因子数据
    output_file = output_path / f"{dataset_name}_daily_factors.csv"
    factors_df.to_csv(output_file, index=False)
    print(f"\n因子数据已保存至: {output_file}")

    # 打印统计摘要
    print(f"\n因子统计摘要 (前5天):")
    print(factors_df.head())

    return factors_df

def visualize_factors(dataset_name, factors_df, config):
    """可视化因子分析结果"""
    print(f"\n生成 {dataset_name} 的可视化图表...")

    # 选择关键因子进行可视化
    key_factors = ['mean', 'std', 'cv', 'trend', 'volatility', 'range']

    # 1. 时间序列图 - 关键因子随时间变化
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'{dataset_name} - Daily Factors Time Series', fontsize=16, fontweight='bold')

    for idx, factor in enumerate(key_factors):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        ax.plot(factors_df['day'], factors_df[factor], linewidth=2, color='steelblue', alpha=0.7)
        ax.fill_between(factors_df['day'], factors_df[factor], alpha=0.3, color='steelblue')
        ax.set_xlabel('Day', fontsize=11)
        ax.set_ylabel(factor.upper(), fontsize=11)
        ax.set_title(f'{factor.upper()} over Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f"{dataset_name}_factors_timeseries.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 分布图 - 因子的分布特征
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'{dataset_name} - Daily Factors Distribution', fontsize=16, fontweight='bold')

    for idx, factor in enumerate(key_factors):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        # 绘制直方图和KDE
        ax.hist(factors_df[factor], bins=30, alpha=0.6, color='skyblue', edgecolor='black', density=True)
        factors_df[factor].plot(kind='kde', ax=ax, color='darkblue', linewidth=2)

        # 添加统计信息
        mean_val = factors_df[factor].mean()
        median_val = factors_df[factor].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')

        ax.set_xlabel(factor.upper(), fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{factor.upper()} Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f"{dataset_name}_factors_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. 相关性热图
    fig, ax = plt.subplots(figsize=(12, 10))

    # 选择数值型因子
    numeric_factors = factors_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_factors.remove('day')  # 移除day列

    corr_matrix = factors_df[numeric_factors].corr()

    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(f'{dataset_name} - Factors Correlation Matrix', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path / f"{dataset_name}_factors_correlation.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. 箱线图 - 因子的离群值分析
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'{dataset_name} - Daily Factors Box Plots', fontsize=16, fontweight='bold')

    for idx, factor in enumerate(key_factors):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        bp = ax.boxplot([factors_df[factor]], labels=[factor.upper()], patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(color='blue', linewidth=1.5),
                        capprops=dict(color='blue', linewidth=1.5))

        ax.set_ylabel('Value', fontsize=11)
        ax.set_title(f'{factor.upper()}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path / f"{dataset_name}_factors_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 已生成 4 张可视化图表")

def generate_summary_report(all_factors):
    """生成所有数据集的汇总报告"""
    print(f"\n{'='*60}")
    print("生成汇总报告")
    print(f"{'='*60}")

    # 创建汇总对比图
    key_factors = ['mean', 'std', 'cv', 'trend', 'volatility', 'range']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('All Datasets - Daily Factors Comparison', fontsize=16, fontweight='bold')

    colors = {'ETTh1': 'steelblue', 'ETTh2': 'coral', 'ETTm1': 'mediumseagreen', 'ETTm2': 'orchid'}

    for idx, factor in enumerate(key_factors):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        for dataset_name, factors_df in all_factors.items():
            ax.plot(factors_df['day'], factors_df[factor],
                   label=dataset_name, linewidth=2, alpha=0.7, color=colors[dataset_name])

        ax.set_xlabel('Day', fontsize=11)
        ax.set_ylabel(factor.upper(), fontsize=11)
        ax.set_title(f'{factor.upper()} Comparison', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "all_datasets_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 生成统计摘要表
    summary_data = []
    for dataset_name, factors_df in all_factors.items():
        for factor in key_factors:
            summary_data.append({
                'Dataset': dataset_name,
                'Factor': factor,
                'Mean': factors_df[factor].mean(),
                'Std': factors_df[factor].std(),
                'Min': factors_df[factor].min(),
                'Max': factors_df[factor].max()
            })

    summary_df = pd.DataFrame(summary_data)
    summary_file = output_path / "all_datasets_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print(f"\n汇总统计表已保存至: {summary_file}")
    print(f"汇总对比图已保存")

def main():
    """主函数"""
    print("="*60)
    print("ETT数据集每日因子计算与可视化")
    print("="*60)

    all_factors = {}

    # 处理每个数据集
    for dataset_name, config in datasets_config.items():
        factors_df = process_dataset(dataset_name, config)
        visualize_factors(dataset_name, factors_df, config)
        all_factors[dataset_name] = factors_df

    # 生成汇总报告
    generate_summary_report(all_factors)

    print(f"\n{'='*60}")
    print("所有处理完成!")
    print(f"结果保存在: {output_path}")
    print(f"{'='*60}")

    # 打印文件清单
    print("\n生成的文件:")
    for file in sorted(output_path.glob("*")):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()
