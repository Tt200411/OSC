import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_daily_volatility(X, observations_per_day):
    """
    计算日内波动率

    σ_d = sqrt( 1/(m-1) * Σ(X_d,i - X̄_d)^2 )

    Args:
        X: 时间序列数据
        observations_per_day: 每天的观测点数量

    Returns:
        volatility_series: 每天的波动率
        daily_segments: 每天的数据段
    """
    n_days = len(X) // observations_per_day
    volatility_series = []
    daily_segments = []

    for d in range(n_days):
        start_idx = d * observations_per_day
        end_idx = start_idx + observations_per_day

        daily_data = X[start_idx:end_idx]
        daily_segments.append(daily_data)

        # 计算样本标准差 (ddof=1)
        sigma_d = np.std(daily_data, ddof=1)
        volatility_series.append(sigma_d)

    return np.array(volatility_series), daily_segments


def calculate_empirical_cdf(data):
    """
    计算经验累积分布函数

    Args:
        data: 数据数组

    Returns:
        sorted_data: 排序后的数据
        cdf_values: 对应的CDF值
    """
    sorted_data = np.sort(data)
    n = len(sorted_data)

    # 使用 (rank) / n 形式
    cdf_values = np.arange(1, n + 1) / n

    return sorted_data, cdf_values


def visualize_volatility_analysis(volatility_series, dataset_name):
    """
    可视化波动率分析结果

    Args:
        volatility_series: 波动率序列
        dataset_name: 数据集名称
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Daily Volatility Analysis - {dataset_name}',
                 fontsize=16, fontweight='bold')

    # 1. 波动率时序图
    axes[0, 0].plot(volatility_series, linewidth=1, alpha=0.7, color='steelblue')
    axes[0, 0].set_title('Daily Volatility - Time Series')
    axes[0, 0].set_xlabel('Day Index')
    axes[0, 0].set_ylabel('Volatility (σ_d)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=np.mean(volatility_series),
                       color='red', linestyle='--',
                       label=f'Mean: {np.mean(volatility_series):.4f}')
    axes[0, 0].legend()

    # 2. 波动率分布直方图
    axes[0, 1].hist(volatility_series, bins=50, alpha=0.7,
                    edgecolor='black', color='steelblue')
    axes[0, 1].set_title('Daily Volatility - Distribution')
    axes[0, 1].set_xlabel('Volatility (σ_d)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=np.mean(volatility_series),
                       color='red', linestyle='--',
                       label=f'Mean: {np.mean(volatility_series):.4f}')
    axes[0, 1].axvline(x=np.median(volatility_series),
                       color='green', linestyle='--',
                       label=f'Median: {np.median(volatility_series):.4f}')
    axes[0, 1].legend()

    # 3. 经验CDF
    sorted_vol, cdf_values = calculate_empirical_cdf(volatility_series)
    axes[1, 0].plot(sorted_vol, cdf_values, linewidth=2, color='darkblue')
    axes[1, 0].set_title('Empirical CDF of Daily Volatility')
    axes[1, 0].set_xlabel('Volatility (σ_d)')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].grid(True, alpha=0.3)

    # 添加分位数线
    quantiles = [0.25, 0.5, 0.75, 0.95]
    colors = ['green', 'orange', 'red', 'purple']
    for q, c in zip(quantiles, colors):
        q_value = np.quantile(volatility_series, q)
        axes[1, 0].axvline(x=q_value, color=c, linestyle='--',
                          alpha=0.6, label=f'Q{int(q*100)}: {q_value:.4f}')
    axes[1, 0].legend()

    # 4. 箱线图
    axes[1, 1].boxplot(volatility_series, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
    axes[1, 1].set_title('Daily Volatility - Box Plot')
    axes[1, 1].set_ylabel('Volatility (σ_d)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # 添加统计信息
    stats_text = f'Mean: {np.mean(volatility_series):.4f}\n'
    stats_text += f'Median: {np.median(volatility_series):.4f}\n'
    stats_text += f'Std: {np.std(volatility_series):.4f}\n'
    stats_text += f'Min: {np.min(volatility_series):.4f}\n'
    stats_text += f'Max: {np.max(volatility_series):.4f}'

    axes[1, 1].text(1.15, np.mean(volatility_series), stats_text,
                    fontsize=10, bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{dataset_name}_volatility_cdf.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_statistics(volatility_series, dataset_name):
    """
    打印统计信息
    """
    print(f"\n{'='*60}")
    print(f"波动率统计摘要 - {dataset_name}")
    print(f"{'='*60}")
    print(f"\n基本统计:")
    print(f"  样本数量: {len(volatility_series)}")
    print(f"  均值:     {np.mean(volatility_series):.6f}")
    print(f"  中位数:   {np.median(volatility_series):.6f}")
    print(f"  标准差:   {np.std(volatility_series):.6f}")
    print(f"  最小值:   {np.min(volatility_series):.6f}")
    print(f"  最大值:   {np.max(volatility_series):.6f}")

    print(f"\n分位数:")
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    for q in quantiles:
        q_value = np.quantile(volatility_series, q)
        print(f"  Q{int(q*100):2d}:      {q_value:.6f}")

    print(f"\n偏度和峰度:")
    from scipy import stats
    skewness = stats.skew(volatility_series)
    kurtosis = stats.kurtosis(volatility_series)
    print(f"  偏度:     {skewness:.6f}")
    print(f"  峰度:     {kurtosis:.6f}")
    print(f"{'='*60}\n")


def analyze_volatility(data_path, dataset_name, observations_per_day):
    """
    完整的波动率分析流程

    Args:
        data_path: 数据文件路径
        dataset_name: 数据集名称
        observations_per_day: 每天的观测点数量
    """
    print(f"\n{'='*60}")
    print(f"开始波动率分析: {dataset_name}")
    print(f"{'='*60}")
    print(f"数据路径: {data_path}")
    print(f"每天观测点数: {observations_per_day}")
    print(f"{'='*60}\n")

    # 加载数据
    df = pd.read_csv(data_path)
    X = df['OT'].values

    print(f"数据加载完成:")
    print(f"  总数据点: {len(X)}")
    print(f"  总天数:   {len(X) // observations_per_day}")
    print(f"  数据范围: [{X.min():.4f}, {X.max():.4f}]\n")

    # 计算日内波动率
    print("计算日内波动率...")
    volatility_series, daily_segments = calculate_daily_volatility(
        X, observations_per_day
    )
    print(f"完成! 共计算了 {len(volatility_series)} 天的波动率\n")

    # 打印统计信息
    print_statistics(volatility_series, dataset_name)

    # 可视化
    print("生成可视化...")
    visualize_volatility_analysis(volatility_series, dataset_name)
    print("完成!\n")

    # 保存结果
    results_df = pd.DataFrame({
        'Day': np.arange(len(volatility_series)),
        'Volatility': volatility_series
    })

    output_path = f'{dataset_name}_daily_volatility.csv'
    results_df.to_csv(output_path, index=False)
    print(f"结果已保存到: {output_path}\n")

    return volatility_series, daily_segments


if __name__ == "__main__":

    # ETTh1 数据集分析 (24点/天)
    print("\n" + "="*60)
    print("ETTh1 数据集 - 日内波动率分析")
    print("="*60)

    vol_h1, segments_h1 = analyze_volatility(
        data_path="COTN/ETT-small/ETTh1.csv",
        dataset_name="ETTh1",
        observations_per_day=24
    )

    # ETTm1 数据集分析 (96点/天)
    print("\n" + "="*60)
    print("ETTm1 数据集 - 日内波动率分析")
    print("="*60)

    vol_m1, segments_m1 = analyze_volatility(
        data_path="COTN/ETT-small/ETTm1.csv",
        dataset_name="ETTm1",
        observations_per_day=96
    )

    print("\n" + "="*60)
    print("所有分析完成!")
    print("="*60)
