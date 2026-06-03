import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# ===============================
# 1. 构造分析段
# ===============================
def construct_segments(X, segment_length, mode="daily"):
    """构造分析段"""
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
    """Rank-based empirical CDF"""
    reference_sorted = np.sort(reference_data)
    n = len(reference_sorted)
    ranks = np.searchsorted(reference_sorted, values, side="right")
    return (ranks + 1) / (n + 1)


# ===============================
# 3. 计算异常密度强度因子
# ===============================
def calculate_anomaly_factor(segments, reference_window_length, alpha=0.10):
    """
    计算异常密度强度因子
    返回每天的异常因子（尾部强度的均值）
    """
    N = len(segments)
    anomaly_factor = np.full(N, np.nan)

    for d in range(reference_window_length, N):
        reference_data = np.concatenate(
            segments[d - reference_window_length:d]
        )

        current_segment = segments[d]

        # 计算经验CDF
        u = empirical_cdf_rank(reference_data, current_segment)
        A = np.minimum(u, 1 - u)

        # 只标记 <= alpha 的异常点
        S_di = np.zeros_like(A)
        mask = A <= alpha

        if np.any(mask):
            A_clipped = np.clip(A[mask], 1e-10, 0.5)
            S_di[mask] = -np.log(2 * A_clipped)

        # 异常因子 = 尾部强度的均值
        anomaly_factor[d] = np.mean(S_di)

    return anomaly_factor


# ===============================
# 4. 计算波动因子
# ===============================
import numpy as np
from scipy.stats import rankdata

def calculate_volatility_factor_full(segments, beta=0.10):
    """
    计算波动因子（Structural Turbulence Component）
    使用整个历史数据拟合分布，平滑连续CDF

    Args:
        segments: list of np.array, 每段的观测数据
        beta: 尾部概率阈值 (默认0.1)

    Returns:
        volatility_factor: np.array，每段波动因子
    """
    N = len(segments)
    volatility_series = np.full(N, np.nan)
    volatility_factor = np.full(N, np.nan)

    # 1. 计算每段的波动率（样本标准差）
    for d in range(N):
        volatility_series[d] = np.std(segments[d], ddof=1)

    # 2. 整体参考分布（全历史段）
    ref_vol = volatility_series[~np.isnan(volatility_series)]
    ref_sorted = np.sort(ref_vol)
    n_ref = len(ref_sorted)

    # 构造平滑连续CDF函数（线性插值）
    def cdf_interp(v):
        return np.interp(v, ref_sorted, np.linspace(1/n_ref, 1, n_ref))

    # 3. 计算波动因子
    for d in range(N):
        V = cdf_interp(volatility_series[d])
        if V >= (1 - beta):
            V_clipped = np.clip(V, 0, 1 - 1e-10)
            volatility_factor[d] = -np.log(1 - V_clipped)
        else:
            volatility_factor[d] = 0.0

    return volatility_factor


# ===============================
# 5. 归一化并计算Turbulence
# ===============================
def calculate_turbulence_normalized(anomaly_factor, volatility_factor, method="sum"):
    """
    归一化两个因子后计算Turbulence

    Args:
        anomaly_factor: 异常因子
        volatility_factor: 波动因子
        method: "sum" 或 "max"

    Returns:
        turbulence: Turbulence因子
        anomaly_normalized: 归一化后的异常因子
        volatility_normalized: 归一化后的波动因子
    """
    # 找到有效值（非NaN）
    valid_mask = ~(np.isnan(anomaly_factor) | np.isnan(volatility_factor))

    # 初始化归一化后的数组
    anomaly_normalized = np.full_like(anomaly_factor, np.nan)
    volatility_normalized = np.full_like(volatility_factor, np.nan)
    turbulence = np.full_like(anomaly_factor, np.nan)

    if np.sum(valid_mask) > 0:
        # 提取有效值
        anomaly_valid = anomaly_factor[valid_mask]
        volatility_valid = volatility_factor[valid_mask]

        # Min-Max归一化到[0, 1]
        anomaly_min, anomaly_max = anomaly_valid.min(), anomaly_valid.max()
        volatility_min, volatility_max = volatility_valid.min(), volatility_valid.max()

        # 避免除以0
        if anomaly_max - anomaly_min > 1e-10:
            anomaly_normalized[valid_mask] = (anomaly_valid - anomaly_min) / (anomaly_max - anomaly_min)
        else:
            anomaly_normalized[valid_mask] = 0.0

        if volatility_max - volatility_min > 1e-10:
            volatility_normalized[valid_mask] = (volatility_valid - volatility_min) / (volatility_max - volatility_min)
        else:
            volatility_normalized[valid_mask] = 0.0

        # 计算Turbulence
        if method == "sum":
            turbulence[valid_mask] = anomaly_normalized[valid_mask] + volatility_normalized[valid_mask]
        elif method == "max":
            turbulence[valid_mask] = np.maximum(
                anomaly_normalized[valid_mask],
                volatility_normalized[valid_mask]
            )

    return turbulence, anomaly_normalized, volatility_normalized


# ===============================
# 6. 可视化时间序列
# ===============================
def visualize_turbulence_timeseries(
        dates,
        anomaly_factor,
        volatility_factor,
        anomaly_normalized,
        volatility_normalized,
        turbulence,
        dataset_name):
    """
    可视化Turbulence时间序列
    """
    fig, axes = plt.subplots(5, 1, figsize=(20, 16))
    fig.suptitle(f'Turbulence Factor Time Series - {dataset_name}',
                 fontsize=16, fontweight='bold')

    valid_mask = ~np.isnan(turbulence)
    valid_indices = np.where(valid_mask)[0]

    # 1. 原始异常因子
    axes[0].plot(valid_indices, anomaly_factor[valid_mask],
                linewidth=1, alpha=0.7, color='steelblue', label='Anomaly Factor')
    axes[0].set_title('Original Anomaly Density Intensity Factor')
    axes[0].set_ylabel('Anomaly Factor')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].axhline(y=np.mean(anomaly_factor[valid_mask]), color='red',
                   linestyle='--', alpha=0.5,
                   label=f'Mean: {np.mean(anomaly_factor[valid_mask]):.4f}')

    # 2. 原始波动因子
    axes[1].plot(valid_indices, volatility_factor[valid_mask],
                linewidth=1, alpha=0.7, color='orange', label='Volatility Factor')
    axes[1].set_title('Original Volatility Factor')
    axes[1].set_ylabel('Volatility Factor')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].axhline(y=np.mean(volatility_factor[valid_mask]), color='red',
                   linestyle='--', alpha=0.5,
                   label=f'Mean: {np.mean(volatility_factor[valid_mask]):.4f}')

    # 3. 归一化后的异常因子
    axes[2].plot(valid_indices, anomaly_normalized[valid_mask],
                linewidth=1, alpha=0.7, color='steelblue', label='Normalized Anomaly')
    axes[2].set_title('Normalized Anomaly Factor [0, 1]')
    axes[2].set_ylabel('Normalized Anomaly')
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    # 4. 归一化后的波动因子
    axes[3].plot(valid_indices, volatility_normalized[valid_mask],
                linewidth=1, alpha=0.7, color='orange', label='Normalized Volatility')
    axes[3].set_title('Normalized Volatility Factor [0, 1]')
    axes[3].set_ylabel('Normalized Volatility')
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    axes[3].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    # 5. Turbulence因子
    axes[4].plot(valid_indices, turbulence[valid_mask],
                linewidth=1.5, alpha=0.8, color='red', label='Turbulence Score')
    axes[4].fill_between(valid_indices, 0, turbulence[valid_mask],
                         alpha=0.3, color='red')
    axes[4].set_title('Turbulence Score (Normalized Anomaly + Normalized Volatility)')
    axes[4].set_xlabel('Day Index')
    axes[4].set_ylabel('Turbulence Score')
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()
    axes[4].axhline(y=np.mean(turbulence[valid_mask]), color='darkred',
                   linestyle='--', alpha=0.7,
                   label=f'Mean: {np.mean(turbulence[valid_mask]):.4f}')

    # 添加高Turbulence区域标记
    threshold = np.percentile(turbulence[valid_mask], 90)
    high_turb_mask = turbulence[valid_mask] > threshold
    if np.any(high_turb_mask):
        axes[4].scatter(valid_indices[high_turb_mask],
                       turbulence[valid_mask][high_turb_mask],
                       color='darkred', s=50, zorder=5,
                       label=f'High Turbulence (>P90={threshold:.2f})')
    axes[4].legend()

    plt.tight_layout()
    plt.savefig(f'{dataset_name}_turbulence_timeseries.png', dpi=300, bbox_inches='tight')
    plt.show()


# ===============================
# 7. 可视化分布对比
# ===============================
def visualize_factor_distributions(
        anomaly_factor,
        volatility_factor,
        anomaly_normalized,
        volatility_normalized,
        turbulence,
        dataset_name):
    """
    可视化因子分布对比
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Factor Distributions - {dataset_name}',
                 fontsize=16, fontweight='bold')

    valid_mask = ~np.isnan(turbulence)

    # 第一行：原始因子分布
    axes[0, 0].hist(anomaly_factor[valid_mask], bins=50, alpha=0.7,
                   edgecolor='black', color='steelblue')
    axes[0, 0].set_title('Original Anomaly Factor')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(volatility_factor[valid_mask], bins=50, alpha=0.7,
                   edgecolor='black', color='orange')
    axes[0, 1].set_title('Original Volatility Factor')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    # 散点图：原始因子关系
    axes[0, 2].scatter(anomaly_factor[valid_mask], volatility_factor[valid_mask],
                      alpha=0.5, s=20, color='purple')
    axes[0, 2].set_title('Original Factors Relationship')
    axes[0, 2].set_xlabel('Anomaly Factor')
    axes[0, 2].set_ylabel('Volatility Factor')
    axes[0, 2].grid(True, alpha=0.3)

    # 计算相关系数
    corr = np.corrcoef(anomaly_factor[valid_mask], volatility_factor[valid_mask])[0, 1]
    axes[0, 2].text(0.05, 0.95, f'Corr: {corr:.3f}',
                   transform=axes[0, 2].transAxes,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 第二行：归一化因子分布
    axes[1, 0].hist(anomaly_normalized[valid_mask], bins=50, alpha=0.7,
                   edgecolor='black', color='steelblue')
    axes[1, 0].set_title('Normalized Anomaly Factor')
    axes[1, 0].set_xlabel('Value [0, 1]')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(volatility_normalized[valid_mask], bins=50, alpha=0.7,
                   edgecolor='black', color='orange')
    axes[1, 1].set_title('Normalized Volatility Factor')
    axes[1, 1].set_xlabel('Value [0, 1]')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)

    # Turbulence分布
    axes[1, 2].hist(turbulence[valid_mask], bins=50, alpha=0.7,
                   edgecolor='black', color='red')
    axes[1, 2].set_title('Turbulence Score Distribution')
    axes[1, 2].set_xlabel('Turbulence Score [0, 2]')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = f'Mean: {np.mean(turbulence[valid_mask]):.3f}\n'
    stats_text += f'Std: {np.std(turbulence[valid_mask]):.3f}\n'
    stats_text += f'Max: {np.max(turbulence[valid_mask]):.3f}'
    axes[1, 2].text(0.95, 0.95, stats_text,
                   transform=axes[1, 2].transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{dataset_name}_factor_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()


# ===============================
# 8. 打印统计摘要
# ===============================
def print_statistics(
        anomaly_factor,
        volatility_factor,
        anomaly_normalized,
        volatility_normalized,
        turbulence,
        dataset_name):
    """打印统计摘要"""
    valid_mask = ~np.isnan(turbulence)

    print(f"\n{'='*60}")
    print(f"Turbulence统计摘要 - {dataset_name}")
    print(f"{'='*60}")

    print(f"\n原始异常因子:")
    print(f"  均值:   {np.mean(anomaly_factor[valid_mask]):.6f}")
    print(f"  中位数: {np.median(anomaly_factor[valid_mask]):.6f}")
    print(f"  标准差: {np.std(anomaly_factor[valid_mask]):.6f}")
    print(f"  最小值: {np.min(anomaly_factor[valid_mask]):.6f}")
    print(f"  最大值: {np.max(anomaly_factor[valid_mask]):.6f}")

    print(f"\n原始波动因子:")
    print(f"  均值:   {np.mean(volatility_factor[valid_mask]):.6f}")
    print(f"  中位数: {np.median(volatility_factor[valid_mask]):.6f}")
    print(f"  标准差: {np.std(volatility_factor[valid_mask]):.6f}")
    print(f"  最小值: {np.min(volatility_factor[valid_mask]):.6f}")
    print(f"  最大值: {np.max(volatility_factor[valid_mask]):.6f}")

    print(f"\n归一化异常因子 [0, 1]:")
    print(f"  均值:   {np.mean(anomaly_normalized[valid_mask]):.6f}")
    print(f"  中位数: {np.median(anomaly_normalized[valid_mask]):.6f}")
    print(f"  标准差: {np.std(anomaly_normalized[valid_mask]):.6f}")

    print(f"\n归一化波动因子 [0, 1]:")
    print(f"  均值:   {np.mean(volatility_normalized[valid_mask]):.6f}")
    print(f"  中位数: {np.median(volatility_normalized[valid_mask]):.6f}")
    print(f"  标准差: {np.std(volatility_normalized[valid_mask]):.6f}")

    print(f"\nTurbulence Score [0, 2]:")
    print(f"  均值:   {np.mean(turbulence[valid_mask]):.6f}")
    print(f"  中位数: {np.median(turbulence[valid_mask]):.6f}")
    print(f"  标准差: {np.std(turbulence[valid_mask]):.6f}")
    print(f"  最小值: {np.min(turbulence[valid_mask]):.6f}")
    print(f"  最大值: {np.max(turbulence[valid_mask]):.6f}")

    print(f"\n分位数:")
    for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        print(f"  Q{int(q*100):2d}: {np.quantile(turbulence[valid_mask], q):.6f}")

    print(f"\n因子相关性:")
    corr = np.corrcoef(anomaly_factor[valid_mask], volatility_factor[valid_mask])[0, 1]
    print(f"  原始因子相关系数: {corr:.6f}")

    corr_norm = np.corrcoef(anomaly_normalized[valid_mask], volatility_normalized[valid_mask])[0, 1]
    print(f"  归一化因子相关系数: {corr_norm:.6f}")

    print(f"{'='*60}\n")


# ===============================
# 9. 主程序
# ===============================
def analyze_turbulence_complete(
        data_path,
        dataset_name,
        observations_per_day,
        reference_window_length=7,
        alpha=0.10,
        beta=0.10,
        mode="daily"):
    """
    完整的Turbulence分析流程
    """
    print(f"\n{'='*60}")
    print(f"完整Turbulence分析 - {dataset_name}")
    print(f"{'='*60}")
    print(f"数据路径: {data_path}")
    print(f"模式: {mode}")
    print(f"每天观测点数: {observations_per_day}")
    print(f"参考窗口长度: {reference_window_length} 天")
    print(f"Alpha (异常阈值): {alpha}")
    print(f"Beta (波动阈值): {beta}")
    print(f"{'='*60}\n")

    # 加载数据
    df = pd.read_csv(data_path)
    X = df['OT'].values

    # 提取日期
    if 'date' in df.columns:
        dates = pd.to_datetime(df['date'])
    else:
        dates = pd.date_range(start='2016-01-01', periods=len(X), freq='H')

    print(f"数据加载完成: {len(X)} 个数据点\n")

    # 1. 构造分段
    print("步骤 1: 构造分段...")
    segments = construct_segments(X, observations_per_day, mode)
    print(f"完成! 共 {len(segments)} 个段\n")

    # 2. 计算异常因子
    print("步骤 2: 计算异常密度强度因子...")
    anomaly_factor = calculate_anomaly_factor(segments, reference_window_length, alpha)
    print("完成!\n")

    # 3. 计算波动因子
    print("步骤 3: 计算波动因子...")
    volatility_factor = calculate_volatility_factor_full(segments, beta)
    print("完成!\n")

    # 4. 归一化并计算Turbulence
    print("步骤 4: 归一化并计算Turbulence...")
    turbulence, anomaly_normalized, volatility_normalized = calculate_turbulence_normalized(
        anomaly_factor, volatility_factor, method="sum"
    )
    print("完成!\n")

    # 5. 打印统计信息
    print_statistics(
        anomaly_factor, volatility_factor,
        anomaly_normalized, volatility_normalized,
        turbulence, dataset_name
    )

    # 6. 可视化时间序列
    print("步骤 5: 生成时间序列可视化...")
    visualize_turbulence_timeseries(
        dates, anomaly_factor, volatility_factor,
        anomaly_normalized, volatility_normalized,
        turbulence, dataset_name
    )
    print("完成!\n")

    # 7. 可视化分布对比
    print("步骤 6: 生成分布对比可视化...")
    visualize_factor_distributions(
        anomaly_factor, volatility_factor,
        anomaly_normalized, volatility_normalized,
        turbulence, dataset_name
    )
    print("完成!\n")

    # 8. 保存结果
    print("步骤 7: 保存结果...")
    results_df = pd.DataFrame({
        'Day': np.arange(len(anomaly_factor)),
        'AnomalyFactor_Original': anomaly_factor,
        'VolatilityFactor_Original': volatility_factor,
        'AnomalyFactor_Normalized': anomaly_normalized,
        'VolatilityFactor_Normalized': volatility_normalized,
        'TurbulenceScore': turbulence
    })

    output_path = f'{dataset_name}_turbulence_complete.csv'
    results_df.to_csv(output_path, index=False)
    print(f"结果已保存到: {output_path}\n")

    return results_df


if __name__ == "__main__":

    # ETTh1 数据集分析
    print("\n" + "="*60)
    print("ETTh1 数据集完整分析")
    print("="*60)

    results_h1 = analyze_turbulence_complete(
        data_path="COTN/ETT-small/ETTh1.csv",
        dataset_name="ETTh1",
        observations_per_day=24,
        reference_window_length=7,
        alpha=0.10,
        beta=0.10,
        mode="daily"
    )

    # ETTm1 数据集分析
    print("\n" + "="*60)
    print("ETTm1 数据集完整分析")
    print("="*60)

    results_m1 = analyze_turbulence_complete(
        data_path="COTN/ETT-small/ETTm1.csv",
        dataset_name="ETTm1",
        observations_per_day=96,
        reference_window_length=7,
        alpha=0.10,
        beta=0.10,
        mode="daily"
    )

    print("\n" + "="*60)
    print("所有分析完成!")
    print("="*60)
