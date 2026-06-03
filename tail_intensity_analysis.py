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
    Rank-based empirical CDF
    使用 (rank+1)/(n+1) 形式
    """
    reference_sorted = np.sort(reference_data)
    n = len(reference_sorted)
    ranks = np.searchsorted(reference_sorted, values, side="right")
    return (ranks + 1) / (n + 1)


# ===============================
# 3. 计算尾部强度因子（简化版）
# ===============================
def calculate_tail_intensity_factor(segments, reference_window_length, alpha=0.05):
    """
    计算尾部强度因子（简化版）
    对于每个段(segment)中的点:
        - 使用前 reference_window_length 天的所有点作为参考
        - 如果点在参考CDF <= alpha 或 >= 1-alpha, 标记为异常点
        - 计算尾部强度 S_di = -log(2*A_di)

    Args:
        segments: 分段列表，每段为 np.array
        reference_window_length: 参考窗口长度（天数/段数）
        alpha: 异常概率阈值

    Returns:
        tail_intensity_list: 每个段的尾部强度数组列表
    """
    N = len(segments)
    tail_intensity_list = [None] * N

    for d in range(reference_window_length, N):
        # 构造参考数据：前 reference_window_length 天的所有点
        reference_data = np.concatenate(
            segments[d - reference_window_length:d]
        )

        current_segment = segments[d]

        # 计算经验CDF
        u = empirical_cdf_rank(reference_data, current_segment)

        # 两尾概率
        A = np.minimum(u, 1 - u)

        # 只标记 <= alpha 的异常点
        S_di = np.zeros_like(A)
        mask = A <= alpha

        # 添加clip避免log(0)
        A_clipped = np.clip(A[mask], 1e-10, 0.5)
        S_di[mask] = -np.log(2 * A_clipped)

        tail_intensity_list[d] = S_di

    return tail_intensity_list


# ===============================
# 4. 可视化分析
# ===============================
def visualize_tail_intensity_analysis(tail_intensity_list, segments, dataset_name, alpha):
    """
    可视化尾部强度分析结果
    """
    # 收集所有有效的尾部强度值
    all_S = []
    all_S_nonzero = []
    anomaly_counts = []

    for d, S_di in enumerate(tail_intensity_list):
        if S_di is not None:
            all_S.extend(S_di)
            all_S_nonzero.extend(S_di[S_di > 0])
            anomaly_counts.append(np.sum(S_di > 0))

    all_S = np.array(all_S)
    all_S_nonzero = np.array(all_S_nonzero)
    anomaly_counts = np.array(anomaly_counts)

    # 创建图表
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    fig.suptitle(f'Tail Intensity Factor Analysis - {dataset_name} (α={alpha})',
                 fontsize=16, fontweight='bold')

    # 1. 所有尾部强度的分布（包括0）
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(all_S, bins=100, alpha=0.7, edgecolor='black', color='steelblue')
    ax1.set_title('All Tail Intensity Values (including zeros)')
    ax1.set_xlabel('S_di')
    ax1.set_ylabel('Frequency')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='S_di = 0')
    ax1.legend()

    # 2. 非零尾部强度的分布
    ax2 = fig.add_subplot(gs[0, 1])
    if len(all_S_nonzero) > 0:
        ax2.hist(all_S_nonzero, bins=80, alpha=0.7, edgecolor='black', color='coral')
        ax2.set_title(f'Non-zero Tail Intensity (n={len(all_S_nonzero)})')
        ax2.set_xlabel('S_di (S_di > 0)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=np.mean(all_S_nonzero), color='red', linestyle='--',
                   label=f'Mean: {np.mean(all_S_nonzero):.3f}')
        ax2.legend()

    # 3. 尾部强度的CDF
    ax3 = fig.add_subplot(gs[0, 2])
    sorted_S = np.sort(all_S_nonzero) if len(all_S_nonzero) > 0 else np.array([0])
    cdf_values = np.arange(1, len(sorted_S) + 1) / len(sorted_S)
    ax3.plot(sorted_S, cdf_values, linewidth=2, color='darkblue')
    ax3.set_title('Empirical CDF of Non-zero Tail Intensity')
    ax3.set_xlabel('S_di')
    ax3.set_ylabel('Cumulative Probability')
    ax3.grid(True, alpha=0.3)

    # 添加分位数线
    if len(all_S_nonzero) > 0:
        quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
        colors = ['green', 'orange', 'red', 'purple', 'brown']
        for q, c in zip(quantiles, colors):
            q_value = np.quantile(all_S_nonzero, q)
            ax3.axvline(x=q_value, color=c, linestyle='--', alpha=0.6,
                       label=f'Q{int(q*100)}: {q_value:.2f}')
        ax3.legend(fontsize=8)

    # 4. 每天异常点数量的时序图
    ax4 = fig.add_subplot(gs[1, :])
    valid_days = [i for i, x in enumerate(tail_intensity_list) if x is not None]
    ax4.plot(valid_days, anomaly_counts, linewidth=1, alpha=0.7, color='darkred')
    ax4.set_title('Number of Anomalous Points per Day')
    ax4.set_xlabel('Day Index')
    ax4.set_ylabel('Count of Anomalous Points')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=np.mean(anomaly_counts), color='blue', linestyle='--',
               label=f'Mean: {np.mean(anomaly_counts):.2f}')
    ax4.legend()

    # 5. 异常点数量的分布
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist(anomaly_counts, bins=30, alpha=0.7, edgecolor='black', color='darkred')
    ax5.set_title('Distribution of Anomalous Point Counts')
    ax5.set_xlabel('Number of Anomalous Points per Day')
    ax5.set_ylabel('Frequency')
    ax5.grid(True, alpha=0.3)

    # 6. 异常比例的分布
    ax6 = fig.add_subplot(gs[2, 1])
    segment_lengths = [len(segments[i]) for i in valid_days]
    anomaly_ratios = anomaly_counts / segment_lengths
    ax6.hist(anomaly_ratios, bins=30, alpha=0.7, edgecolor='black', color='purple')
    ax6.set_title('Distribution of Anomaly Ratios')
    ax6.set_xlabel('Anomaly Ratio (anomalies / segment_length)')
    ax6.set_ylabel('Frequency')
    ax6.grid(True, alpha=0.3)
    ax6.axvline(x=alpha, color='red', linestyle='--', linewidth=2,
               label=f'Expected α={alpha}')
    ax6.legend()

    # 7. 统计摘要文本
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    stats_text = "Statistical Summary\n" + "="*40 + "\n\n"
    stats_text += f"Total data points: {len(all_S)}\n"
    stats_text += f"Anomalous points: {len(all_S_nonzero)}\n"
    stats_text += f"Anomaly ratio: {len(all_S_nonzero)/len(all_S)*100:.2f}%\n"
    stats_text += f"Expected ratio: {alpha*100:.2f}%\n\n"

    if len(all_S_nonzero) > 0:
        stats_text += "Non-zero Tail Intensity:\n"
        stats_text += f"  Mean:   {np.mean(all_S_nonzero):.4f}\n"
        stats_text += f"  Median: {np.median(all_S_nonzero):.4f}\n"
        stats_text += f"  Std:    {np.std(all_S_nonzero):.4f}\n"
        stats_text += f"  Min:    {np.min(all_S_nonzero):.4f}\n"
        stats_text += f"  Max:    {np.max(all_S_nonzero):.4f}\n\n"

    stats_text += "Anomalous Points per Day:\n"
    stats_text += f"  Mean:   {np.mean(anomaly_counts):.2f}\n"
    stats_text += f"  Median: {np.median(anomaly_counts):.2f}\n"
    stats_text += f"  Std:    {np.std(anomaly_counts):.2f}\n"
    stats_text += f"  Min:    {np.min(anomaly_counts):.0f}\n"
    stats_text += f"  Max:    {np.max(anomaly_counts):.0f}\n"

    ax7.text(0.1, 0.9, stats_text, transform=ax7.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.savefig(f'{dataset_name}_tail_intensity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印详细统计
    print(f"\n{'='*60}")
    print(f"尾部强度统计摘要 - {dataset_name}")
    print(f"{'='*60}")
    print(f"\n总数据点: {len(all_S)}")
    print(f"异常点数量: {len(all_S_nonzero)}")
    print(f"异常比例: {len(all_S_nonzero)/len(all_S)*100:.4f}%")
    print(f"期望比例 (α): {alpha*100:.2f}%")

    if len(all_S_nonzero) > 0:
        print(f"\n非零尾部强度统计:")
        print(f"  均值:   {np.mean(all_S_nonzero):.6f}")
        print(f"  中位数: {np.median(all_S_nonzero):.6f}")
        print(f"  标准差: {np.std(all_S_nonzero):.6f}")
        print(f"  最小值: {np.min(all_S_nonzero):.6f}")
        print(f"  最大值: {np.max(all_S_nonzero):.6f}")

        print(f"\n分位数:")
        for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
            print(f"  Q{int(q*100):2d}: {np.quantile(all_S_nonzero, q):.6f}")

    print(f"\n每天异常点数量:")
    print(f"  均值:   {np.mean(anomaly_counts):.4f}")
    print(f"  中位数: {np.median(anomaly_counts):.4f}")
    print(f"  标准差: {np.std(anomaly_counts):.4f}")
    print(f"  最小值: {np.min(anomaly_counts):.0f}")
    print(f"  最大值: {np.max(anomaly_counts):.0f}")
    print(f"{'='*60}\n")


# ===============================
# 5. OT概率分布可视化
# ===============================
def visualize_ot_distribution(X, dataset_name):
    """
    可视化OT的概率分布
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'OT Probability Distribution - {dataset_name}',
                 fontsize=16, fontweight='bold')

    # 1. 直方图
    axes[0, 0].hist(X, bins=100, alpha=0.7, edgecolor='black',
                    color='steelblue', density=True)
    axes[0, 0].set_title('Histogram (Normalized)')
    axes[0, 0].set_xlabel('OT Value')
    axes[0, 0].set_ylabel('Probability Density')
    axes[0, 0].grid(True, alpha=0.3)

    # 添加统计线
    axes[0, 0].axvline(x=np.mean(X), color='red', linestyle='--',
                      label=f'Mean: {np.mean(X):.2f}')
    axes[0, 0].axvline(x=np.median(X), color='green', linestyle='--',
                      label=f'Median: {np.median(X):.2f}')
    axes[0, 0].legend()

    # 2. 核密度估计
    axes[0, 1].hist(X, bins=100, alpha=0.3, edgecolor='black',
                    color='steelblue', density=True, label='Histogram')

    # KDE
    from scipy import stats
    kde = stats.gaussian_kde(X)
    x_range = np.linspace(X.min(), X.max(), 1000)
    axes[0, 1].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    axes[0, 1].set_title('Kernel Density Estimation')
    axes[0, 1].set_xlabel('OT Value')
    axes[0, 1].set_ylabel('Probability Density')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. 经验CDF
    sorted_X = np.sort(X)
    cdf_values = np.arange(1, len(sorted_X) + 1) / len(sorted_X)
    axes[1, 0].plot(sorted_X, cdf_values, linewidth=2, color='darkblue')
    axes[1, 0].set_title('Empirical Cumulative Distribution Function')
    axes[1, 0].set_xlabel('OT Value')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].grid(True, alpha=0.3)

    # 添加分位数线
    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    colors = ['purple', 'green', 'orange', 'red', 'brown']
    for q, c in zip(quantiles, colors):
        q_value = np.quantile(X, q)
        axes[1, 0].axvline(x=q_value, color=c, linestyle='--', alpha=0.6,
                          label=f'Q{int(q*100)}: {q_value:.2f}')
    axes[1, 0].legend(fontsize=8)

    # 4. Q-Q图 (与正态分布比较)
    from scipy import stats
    stats.probplot(X, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (vs Normal Distribution)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{dataset_name}_ot_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"OT分布统计 - {dataset_name}")
    print(f"{'='*60}")
    print(f"\n基本统计:")
    print(f"  样本数量: {len(X)}")
    print(f"  均值:     {np.mean(X):.6f}")
    print(f"  中位数:   {np.median(X):.6f}")
    print(f"  标准差:   {np.std(X):.6f}")
    print(f"  最小值:   {np.min(X):.6f}")
    print(f"  最大值:   {np.max(X):.6f}")

    print(f"\n分位数:")
    for q in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]:
        print(f"  Q{int(q*100):2d}: {np.quantile(X, q):.6f}")

    print(f"\n偏度和峰度:")
    skewness = stats.skew(X)
    kurtosis = stats.kurtosis(X)
    print(f"  偏度:     {skewness:.6f}")
    print(f"  峰度:     {kurtosis:.6f}")
    print(f"{'='*60}\n")


# ===============================
# 6. 主程序
# ===============================
def analyze_tail_intensity(
        data_path,
        dataset_name,
        observations_per_day,
        reference_window_length=7,
        alpha=0.05,
        mode="daily"):
    """
    完整的尾部强度分析流程
    """
    print(f"\n{'='*60}")
    print(f"尾部强度因子分析 - {dataset_name}")
    print(f"{'='*60}")
    print(f"数据路径: {data_path}")
    print(f"模式: {mode}")
    print(f"每天观测点数: {observations_per_day}")
    print(f"参考窗口长度: {reference_window_length} 天")
    print(f"Alpha (异常阈值): {alpha}")
    print(f"{'='*60}\n")

    # 加载数据
    df = pd.read_csv(data_path)
    X = df['OT'].values

    print(f"数据加载完成:")
    print(f"  总数据点: {len(X)}")
    print(f"  数据范围: [{X.min():.4f}, {X.max():.4f}]\n")

    # 1. 可视化OT的概率分布
    print("="*60)
    print("步骤 1: 分析OT的概率分布")
    print("="*60)
    visualize_ot_distribution(X, dataset_name)

    # 2. 构造分段
    print("="*60)
    print("步骤 2: 构造分段")
    print("="*60)
    segments = construct_segments(X, observations_per_day, mode)
    print(f"分段完成: {len(segments)} 个段\n")

    # 3. 计算尾部强度因子
    print("="*60)
    print("步骤 3: 计算尾部强度因子")
    print("="*60)
    tail_intensity_list = calculate_tail_intensity_factor(
        segments, reference_window_length, alpha
    )
    print("计算完成!\n")

    # 4. 可视化尾部强度分析
    print("="*60)
    print("步骤 4: 可视化尾部强度分析")
    print("="*60)
    visualize_tail_intensity_analysis(
        tail_intensity_list, segments, dataset_name, alpha
    )

    # 5. 保存结果
    print("="*60)
    print("步骤 5: 保存结果")
    print("="*60)

    # 保存每天的尾部强度数据
    results = []
    for d, S_di in enumerate(tail_intensity_list):
        if S_di is not None:
            for i, s_value in enumerate(S_di):
                results.append({
                    'day': d,
                    'point_index': i,
                    'tail_intensity': s_value,
                    'is_anomaly': s_value > 0
                })

    results_df = pd.DataFrame(results)
    output_path = f'{dataset_name}_tail_intensity_results.csv'
    results_df.to_csv(output_path, index=False)
    print(f"结果已保存到: {output_path}\n")

    return tail_intensity_list, segments


if __name__ == "__main__":

    # ETTh1 数据集分析
    print("\n" + "="*60)
    print("ETTh1 数据集分析")
    print("="*60)

    tail_h1, segments_h1 = analyze_tail_intensity(
        data_path="COTN/ETT-small/ETTh1.csv",
        dataset_name="ETTh1",
        observations_per_day=24,
        reference_window_length=7,
        alpha=0.05,
        mode="daily"
    )

    # ETTm1 数据集分析
    print("\n" + "="*60)
    print("ETTm1 数据集分析")
    print("="*60)

    tail_m1, segments_m1 = analyze_tail_intensity(
        data_path="COTN/ETT-small/ETTm1.csv",
        dataset_name="ETTm1",
        observations_per_day=96,
        reference_window_length=7,
        alpha=0.05,
        mode="daily"
    )

    print("\n" + "="*60)
    print("所有分析完成!")
    print("="*60)
