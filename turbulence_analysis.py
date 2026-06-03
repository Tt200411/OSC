import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def construct_segments(X, segment_length, mode):
    """
    构造参考窗口

    Args:
        X: 时间序列数据
        segment_length: 段长度
        mode: "daily" 或 "rolling"

    Returns:
        segments: 分段列表
    """
    segments = []

    if mode == "daily":
        step = segment_length
    else:  # rolling
        step = 1

    for t in range(0, len(X) - segment_length + 1, step):
        segment = X[t : t + segment_length]
        segments.append(segment)

    return segments


def calculate_anomaly_density_intensity(segments, reference_window_length, alpha):
    """
    计算异常密度强度因子 (Distributional Turbulence Component)

    Args:
        segments: 分段列表
        reference_window_length: 参考窗口长度
        alpha: 异常阈值

    Returns:
        anomaly_factor: 异常密度强度因子数组
    """
    N = len(segments)
    anomaly_factor = np.full(N, np.nan)

    for d in range(reference_window_length, N):
        # 构造参考数据
        reference_data = np.concatenate(
            segments[d - reference_window_length : d]
        )

        # 计算经验CDF
        def empirical_cdf(value):
            return np.sum(reference_data <= value) / len(reference_data)

        anomaly_ratio = 0
        tail_intensity = 0

        current_segment = segments[d]

        for value in current_segment:
            u = empirical_cdf(value)
            A = min(u, 1 - u)

            if A <= alpha:
                anomaly_ratio += 1
                tail_intensity += -np.log(2 * A + 1e-10)  # 避免log(0)

        # 密度成分
        density_component = anomaly_ratio / len(current_segment)

        # 强度成分
        intensity_component = tail_intensity / len(current_segment)

        # 异常因子 = 密度 + 强度
        anomaly_factor[d] = density_component + intensity_component

    return anomaly_factor


def calculate_volatility_factor(segments, reference_window_length, beta):
    """
    计算波动因子 (Structural Turbulence Component)

    Args:
        segments: 分段列表
        reference_window_length: 参考窗口长度
        beta: 波动阈值

    Returns:
        volatility_factor: 波动因子数组
    """
    N = len(segments)
    volatility_series = np.zeros(N)
    volatility_factor = np.full(N, np.nan)

    # 1. 计算每个segment的波动率
    for d in range(N):
        volatility_series[d] = np.std(segments[d])

    # 2. 滑动分布判断
    for d in range(reference_window_length, N):
        reference_vol = volatility_series[d - reference_window_length : d]

        # 计算波动率的经验CDF
        def vol_cdf(v):
            return np.sum(reference_vol <= v) / len(reference_vol)

        V = vol_cdf(volatility_series[d])

        if V >= (1 - beta):
            volatility_factor[d] = -np.log(1 - V + 1e-10)  # 避免log(0)
        else:
            volatility_factor[d] = 0

    return volatility_factor


def calculate_turbulence(anomaly_factor, volatility_factor, method="sum"):
    """
    计算湍流分数

    Args:
        anomaly_factor: 异常密度强度因子
        volatility_factor: 波动因子
        method: "sum" 或 "max"

    Returns:
        turbulence: 湍流分数数组
    """
    N = len(anomaly_factor)
    turbulence = np.zeros(N)

    for d in range(N):
        if method == "sum":
            turbulence[d] = anomaly_factor[d] + volatility_factor[d]
        elif method == "max":
            turbulence[d] = max(anomaly_factor[d], volatility_factor[d])

    return turbulence


def visualize_all(anomaly_factor, volatility_factor, turbulence, dataset_name):
    """
    可视化所有结果

    Args:
        anomaly_factor: 异常密度强度因子
        volatility_factor: 波动因子
        turbulence: 湍流分数
        dataset_name: 数据集名称
    """
    # 过滤掉NaN值
    anomaly_valid = anomaly_factor[~np.isnan(anomaly_factor)]
    volatility_valid = volatility_factor[~np.isnan(volatility_factor)]
    turbulence_valid = turbulence[~np.isnan(turbulence)]

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Turbulence Analysis - {dataset_name}', fontsize=16, fontweight='bold')

    # 1. Anomaly Factor - 时序图
    axes[0, 0].plot(anomaly_valid, linewidth=1, alpha=0.7)
    axes[0, 0].set_title('Anomaly Density Intensity Factor - Time Series')
    axes[0, 0].set_xlabel('Segment Index')
    axes[0, 0].set_ylabel('Anomaly Factor')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Anomaly Factor - 分布
    axes[0, 1].hist(anomaly_valid, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Anomaly Density Intensity Factor - Distribution')
    axes[0, 1].set_xlabel('Anomaly Factor')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Volatility Factor - 时序图
    axes[1, 0].plot(volatility_valid, linewidth=1, alpha=0.7, color='orange')
    axes[1, 0].set_title('Volatility Factor - Time Series')
    axes[1, 0].set_xlabel('Segment Index')
    axes[1, 0].set_ylabel('Volatility Factor')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Volatility Factor - 分布
    axes[1, 1].hist(volatility_valid, bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 1].set_title('Volatility Factor - Distribution')
    axes[1, 1].set_xlabel('Volatility Factor')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Turbulence Score - 时序图
    axes[2, 0].plot(turbulence_valid, linewidth=1, alpha=0.7, color='red')
    axes[2, 0].set_title('Turbulence Score - Time Series')
    axes[2, 0].set_xlabel('Segment Index')
    axes[2, 0].set_ylabel('Turbulence Score')
    axes[2, 0].grid(True, alpha=0.3)

    # 6. Turbulence Score - 分布
    axes[2, 1].hist(turbulence_valid, bins=50, alpha=0.7, edgecolor='black', color='red')
    axes[2, 1].set_title('Turbulence Score - Distribution')
    axes[2, 1].set_xlabel('Turbulence Score')
    axes[2, 1].set_ylabel('Frequency')
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{dataset_name}_turbulence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n{'='*60}")
    print(f"统计摘要 - {dataset_name}")
    print(f"{'='*60}")
    print(f"\nAnomaly Density Intensity Factor:")
    print(f"  Mean: {np.mean(anomaly_valid):.4f}")
    print(f"  Std:  {np.std(anomaly_valid):.4f}")
    print(f"  Min:  {np.min(anomaly_valid):.4f}")
    print(f"  Max:  {np.max(anomaly_valid):.4f}")

    print(f"\nVolatility Factor:")
    print(f"  Mean: {np.mean(volatility_valid):.4f}")
    print(f"  Std:  {np.std(volatility_valid):.4f}")
    print(f"  Min:  {np.min(volatility_valid):.4f}")
    print(f"  Max:  {np.max(volatility_valid):.4f}")

    print(f"\nTurbulence Score:")
    print(f"  Mean: {np.mean(turbulence_valid):.4f}")
    print(f"  Std:  {np.std(turbulence_valid):.4f}")
    print(f"  Min:  {np.min(turbulence_valid):.4f}")
    print(f"  Max:  {np.max(turbulence_valid):.4f}")
    print(f"{'='*60}\n")


def main(data_path, dataset_name, observations_per_day,
         reference_window_length=7, alpha=0.05, beta=0.05,
         mode="daily", method="sum"):
    """
    主程序

    Args:
        data_path: 数据文件路径
        dataset_name: 数据集名称
        observations_per_day: 每天的观测点数量
        reference_window_length: 参考窗口长度（天数）
        alpha: 异常阈值
        beta: 波动阈值
        mode: "daily" 或 "rolling"
        method: "sum" 或 "max"
    """
    print(f"\n{'='*60}")
    print(f"开始分析: {dataset_name}")
    print(f"{'='*60}")
    print(f"模式: {mode}")
    print(f"每天观测点数: {observations_per_day}")
    print(f"参考窗口长度: {reference_window_length} 天")
    print(f"Alpha (异常阈值): {alpha}")
    print(f"Beta (波动阈值): {beta}")
    print(f"湍流计算方法: {method}")
    print(f"{'='*60}\n")

    # 1. 加载数据
    df = pd.read_csv(data_path)
    X = df['OT'].values

    print(f"数据加载完成: {len(X)} 个数据点")
    print(f"数据范围: {X.min():.4f} ~ {X.max():.4f}\n")

    # 2. 构造分段
    segment_length = observations_per_day
    segments = construct_segments(X, segment_length, mode)
    print(f"分段完成: {len(segments)} 个段\n")

    # 3. 计算异常密度强度因子
    print("计算异常密度强度因子...")
    anomaly_factor = calculate_anomaly_density_intensity(
        segments, reference_window_length, alpha
    )
    print("完成!\n")

    # 4. 计算波动因子
    print("计算波动因子...")
    volatility_factor = calculate_volatility_factor(
        segments, reference_window_length, beta
    )
    print("完成!\n")

    # 5. 计算湍流分数
    print("计算湍流分数...")
    turbulence_score = calculate_turbulence(
        anomaly_factor, volatility_factor, method
    )
    print("完成!\n")

    # 6. 可视化
    print("生成可视化...")
    visualize_all(anomaly_factor, volatility_factor, turbulence_score, dataset_name)
    print("完成!\n")

    # 7. 保存结果
    results_df = pd.DataFrame({
        'AnomalyDensityIntensityFactor': anomaly_factor,
        'VolatilityFactor': volatility_factor,
        'TurbulenceScore': turbulence_score
    })

    output_path = f'{dataset_name}_turbulence_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"结果已保存到: {output_path}\n")

    return {
        "AnomalyDensityIntensityFactor": anomaly_factor,
        "VolatilityFactor": volatility_factor,
        "TurbulenceScore": turbulence_score
    }


if __name__ == "__main__":
    # ETTh1 数据集分析
    results_h1 = main(
        data_path="COTN/ETT-small/ETTh1.csv",
        dataset_name="ETTh1",
        observations_per_day=24,
        reference_window_length=7,
        alpha=0.05,
        beta=0.05,
        mode="daily",
        method="sum"
    )

    # ETTm1 数据集分析
    results_m1 = main(
        data_path="COTN/ETT-small/ETTm1.csv",
        dataset_name="ETTm1",
        observations_per_day=96,
        reference_window_length=7,
        alpha=0.05,
        beta=0.05,
        mode="daily",
        method="sum"
    )
