import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ===============================
# 1. 构造分析段
# ===============================
def construct_segments(X, segment_length, mode="daily"):
    """
    构造分析段

    mode:
        "daily"   -> 非重叠段
        "rolling" -> 滑动窗口段
    """
    segments = []

    if mode == "daily":
        step = segment_length
    else:
        step = 1

    for t in range(0, len(X) - segment_length + 1, step):
        segments.append(X[t:t + segment_length])

    return segments


# ===============================
# 2. Rank-based 经验CDF
# ===============================
def empirical_cdf_rank(reference_data, values):
    """
    使用 rank/(n+1) 形式计算经验CDF
    向量化实现
    """
    reference_sorted = np.sort(reference_data)
    n = len(reference_sorted)

    # searchsorted 返回插入位置
    ranks = np.searchsorted(reference_sorted, values, side="right")

    return (ranks + 1) / (n + 1)


# ===============================
# 3. 异常密度强度因子
# ===============================
def calculate_anomaly_density_intensity(
        segments,
        reference_window_length,
        alpha,
        normalize_intensity=True):

    N = len(segments)
    anomaly_factor = np.full(N, np.nan)

    max_theoretical_intensity = -np.log(2 * alpha)

    for d in range(reference_window_length, N):

        reference_data = np.concatenate(
            segments[d - reference_window_length:d]
        )

        current_segment = segments[d]

        # 计算经验CDF
        u = empirical_cdf_rank(reference_data, current_segment)

        A = np.minimum(u, 1 - u)

        # 密度成分
        anomaly_mask = A <= alpha
        density_component = np.mean(anomaly_mask)

        # 强度成分
        # 添加小的epsilon避免log(0)
        A_clipped = np.clip(A[anomaly_mask], 1e-10, 0.5)
        tail_intensity = np.sum(-np.log(2 * A_clipped))
        intensity_component = tail_intensity / len(current_segment)

        # 理论标准化（推荐）
        if normalize_intensity and max_theoretical_intensity > 0:
            intensity_component = intensity_component / max_theoretical_intensity

        anomaly_factor[d] = density_component + intensity_component

    return anomaly_factor


# ===============================
# 4. 波动因子
# ===============================
def calculate_volatility_factor(
        segments,
        reference_window_length,
        beta):

    N = len(segments)
    volatility_series = np.full(N, np.nan)
    volatility_factor = np.full(N, np.nan)

    # 使用样本标准差 ddof=1
    for d in range(N):
        volatility_series[d] = np.std(segments[d], ddof=1)

    for d in range(reference_window_length, N):

        reference_vol = volatility_series[d - reference_window_length:d]

        # rank-based CDF
        V = empirical_cdf_rank(reference_vol, [volatility_series[d]])[0]

        if V >= (1 - beta):
            # 添加小的epsilon避免log(0)
            V_clipped = np.clip(V, 0, 1 - 1e-10)
            volatility_factor[d] = -np.log(1 - V_clipped)
        else:
            volatility_factor[d] = 0.0

    return volatility_factor


# ===============================
# 5. Turbulence 计算
# ===============================
def calculate_turbulence(
        anomaly_factor,
        volatility_factor,
        method="sum"):

    N = len(anomaly_factor)
    turbulence = np.full(N, np.nan)

    for d in range(N):

        if np.isnan(anomaly_factor[d]) or np.isnan(volatility_factor[d]):
            continue

        if method == "sum":
            turbulence[d] = anomaly_factor[d] + volatility_factor[d]
        elif method == "max":
            turbulence[d] = max(
                anomaly_factor[d],
                volatility_factor[d]
            )
        else:
            raise ValueError("method must be 'sum' or 'max'")

    return turbulence


# ===============================
# 6. 可视化
# ===============================
def visualize_results(
        anomaly_factor,
        volatility_factor,
        turbulence,
        dataset_name):

    valid = ~np.isnan(turbulence)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f"Turbulence Analysis - {dataset_name}", fontsize=16, fontweight='bold')

    # Anomaly Factor
    axes[0, 0].plot(anomaly_factor[valid], linewidth=1, alpha=0.7)
    axes[0, 0].set_title("Anomaly Density Intensity Factor (Time Series)")
    axes[0, 0].set_xlabel("Segment Index")
    axes[0, 0].set_ylabel("Anomaly Factor")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(anomaly_factor[valid], bins=40, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title("Anomaly Density Intensity Factor (Distribution)")
    axes[0, 1].set_xlabel("Anomaly Factor")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].grid(True, alpha=0.3)

    # Volatility Factor
    axes[1, 0].plot(volatility_factor[valid], linewidth=1, alpha=0.7, color='orange')
    axes[1, 0].set_title("Volatility Factor (Time Series)")
    axes[1, 0].set_xlabel("Segment Index")
    axes[1, 0].set_ylabel("Volatility Factor")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(volatility_factor[valid], bins=40, alpha=0.7, edgecolor='black', color='orange')
    axes[1, 1].set_title("Volatility Factor (Distribution)")
    axes[1, 1].set_xlabel("Volatility Factor")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3)

    # Turbulence Score
    axes[2, 0].plot(turbulence[valid], linewidth=1, alpha=0.7, color='red')
    axes[2, 0].set_title("Turbulence Score (Time Series)")
    axes[2, 0].set_xlabel("Segment Index")
    axes[2, 0].set_ylabel("Turbulence Score")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].hist(turbulence[valid], bins=40, alpha=0.7, edgecolor='black', color='red')
    axes[2, 1].set_title("Turbulence Score (Distribution)")
    axes[2, 1].set_xlabel("Turbulence Score")
    axes[2, 1].set_ylabel("Frequency")
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{dataset_name}_turbulence_v2.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"统计摘要 - {dataset_name}")
    print(f"{'='*60}")

    anomaly_valid = anomaly_factor[valid]
    volatility_valid = volatility_factor[valid]
    turbulence_valid = turbulence[valid]

    print(f"\nAnomaly Density Intensity Factor:")
    print(f"  Mean:   {np.mean(anomaly_valid):.6f}")
    print(f"  Std:    {np.std(anomaly_valid):.6f}")
    print(f"  Min:    {np.min(anomaly_valid):.6f}")
    print(f"  Median: {np.median(anomaly_valid):.6f}")
    print(f"  Max:    {np.max(anomaly_valid):.6f}")

    print(f"\nVolatility Factor:")
    print(f"  Mean:   {np.mean(volatility_valid):.6f}")
    print(f"  Std:    {np.std(volatility_valid):.6f}")
    print(f"  Min:    {np.min(volatility_valid):.6f}")
    print(f"  Median: {np.median(volatility_valid):.6f}")
    print(f"  Max:    {np.max(volatility_valid):.6f}")

    print(f"\nTurbulence Score:")
    print(f"  Mean:   {np.mean(turbulence_valid):.6f}")
    print(f"  Std:    {np.std(turbulence_valid):.6f}")
    print(f"  Min:    {np.min(turbulence_valid):.6f}")
    print(f"  Median: {np.median(turbulence_valid):.6f}")
    print(f"  Max:    {np.max(turbulence_valid):.6f}")
    print(f"{'='*60}\n")


# ===============================
# 7. 主程序
# ===============================
def run_turbulence_analysis(
        X,
        observations_per_day,
        reference_window_length=7,
        alpha=0.05,
        beta=0.05,
        mode="daily",
        method="sum",
        normalize_intensity=True):

    segments = construct_segments(
        X,
        observations_per_day,
        mode
    )

    anomaly_factor = calculate_anomaly_density_intensity(
        segments,
        reference_window_length,
        alpha,
        normalize_intensity
    )

    volatility_factor = calculate_volatility_factor(
        segments,
        reference_window_length,
        beta
    )

    turbulence = calculate_turbulence(
        anomaly_factor,
        volatility_factor,
        method
    )

    return anomaly_factor, volatility_factor, turbulence


# ===============================
# 8. 完整分析流程
# ===============================
def analyze_dataset(
        data_path,
        dataset_name,
        observations_per_day,
        reference_window_length=7,
        alpha=0.05,
        beta=0.05,
        mode="daily",
        method="sum",
        normalize_intensity=True):
    """
    完整的数据集分析流程
    """
    print(f"\n{'='*60}")
    print(f"开始分析: {dataset_name}")
    print(f"{'='*60}")
    print(f"数据路径: {data_path}")
    print(f"模式: {mode}")
    print(f"每天观测点数: {observations_per_day}")
    print(f"参考窗口长度: {reference_window_length} 天")
    print(f"Alpha (异常阈值): {alpha}")
    print(f"Beta (波动阈值): {beta}")
    print(f"湍流计算方法: {method}")
    print(f"强度标准化: {normalize_intensity}")
    print(f"{'='*60}\n")

    # 加载数据
    df = pd.read_csv(data_path)
    X = df['OT'].values

    print(f"数据加载完成:")
    print(f"  总数据点: {len(X)}")
    print(f"  数据范围: [{X.min():.4f}, {X.max():.4f}]")
    print(f"  数据均值: {X.mean():.4f}")
    print(f"  数据标准差: {X.std():.4f}\n")

    # 运行分析
    print("开始计算湍流因子...\n")
    anomaly_factor, volatility_factor, turbulence = run_turbulence_analysis(
        X,
        observations_per_day,
        reference_window_length,
        alpha,
        beta,
        mode,
        method,
        normalize_intensity
    )

    print("计算完成!\n")

    # 可视化
    print("生成可视化...")
    visualize_results(anomaly_factor, volatility_factor, turbulence, dataset_name)

    # 保存结果
    results_df = pd.DataFrame({
        'AnomalyDensityIntensityFactor': anomaly_factor,
        'VolatilityFactor': volatility_factor,
        'TurbulenceScore': turbulence
    })

    output_path = f'{dataset_name}_turbulence_v2.csv'
    results_df.to_csv(output_path, index=False)
    print(f"结果已保存到: {output_path}\n")

    return {
        "AnomalyDensityIntensityFactor": anomaly_factor,
        "VolatilityFactor": volatility_factor,
        "TurbulenceScore": turbulence
    }


if __name__ == "__main__":

    # ETTh1 数据集分析 (24点/天)
    print("\n" + "="*60)
    print("ETTh1 数据集分析")
    print("="*60)

    results_h1 = analyze_dataset(
        data_path="COTN/ETT-small/ETTh1.csv",
        dataset_name="ETTh1",
        observations_per_day=24,
        reference_window_length=30,
        alpha=0.05,
        beta=0.05,
        mode="daily",
        method="sum",
        normalize_intensity=True
    )

    # ETTm1 数据集分析 (96点/天)
    print("\n" + "="*60)
    print("ETTm1 数据集分析")
    print("="*60)

    results_m1 = analyze_dataset(
        data_path="COTN/ETT-small/ETTm1.csv",
        dataset_name="ETTm1",
        observations_per_day=96,
        reference_window_length=30,
        alpha=0.05,
        beta=0.05,
        mode="daily",
        method="sum",
        normalize_intensity=True
    )

    print("\n" + "="*60)
    print("所有分析完成!")
    print("="*60)
