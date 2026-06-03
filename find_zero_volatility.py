import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_zero_volatility_days(data_path, dataset_name, observations_per_day):
    """
    找出波动率为0的日期，并分析原因
    """
    print(f"\n{'='*60}")
    print(f"查找波动率为0的日期 - {dataset_name}")
    print(f"{'='*60}\n")

    # 加载数据
    df = pd.read_csv(data_path)
    X = df['OT'].values

    n_days = len(X) // observations_per_day
    zero_vol_days = []
    zero_vol_details = []

    for d in range(n_days):
        start_idx = d * observations_per_day
        end_idx = start_idx + observations_per_day

        daily_data = X[start_idx:end_idx]

        # 计算样本标准差
        sigma_d = np.std(daily_data, ddof=1)

        if sigma_d == 0.0 or np.isclose(sigma_d, 0.0, atol=1e-10):
            zero_vol_days.append(d)

            # 获取日期信息
            if 'date' in df.columns:
                date_str = df.iloc[start_idx]['date']
            else:
                date_str = f"Day {d}"

            # 分析原因
            unique_values = np.unique(daily_data)
            all_same = len(unique_values) == 1

            zero_vol_details.append({
                'day_index': d,
                'date': date_str,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'volatility': sigma_d,
                'mean': np.mean(daily_data),
                'min': np.min(daily_data),
                'max': np.max(daily_data),
                'unique_values': len(unique_values),
                'all_same': all_same,
                'sample_values': daily_data[:5].tolist()
            })

    print(f"总天数: {n_days}")
    print(f"波动率为0的天数: {len(zero_vol_days)}")
    print(f"占比: {len(zero_vol_days)/n_days*100:.2f}%\n")

    if len(zero_vol_days) > 0:
        print(f"{'='*60}")
        print("波动率为0的日期详情:")
        print(f"{'='*60}\n")

        for i, detail in enumerate(zero_vol_details, 1):
            print(f"[{i}] Day {detail['day_index']} - {detail['date']}")
            print(f"    数据索引: [{detail['start_idx']}, {detail['end_idx']})")
            print(f"    波动率: {detail['volatility']:.10f}")
            print(f"    均值: {detail['mean']:.6f}")
            print(f"    范围: [{detail['min']:.6f}, {detail['max']:.6f}]")
            print(f"    唯一值数量: {detail['unique_values']}")
            print(f"    所有值相同: {detail['all_same']}")
            print(f"    前5个值: {detail['sample_values']}")

            if detail['all_same']:
                print(f"    ⚠️  原因: 该天所有{observations_per_day}个数据点的值完全相同!")
            else:
                print(f"    ⚠️  原因: 该天数据变化极小，标准差接近0")
            print()

        # 可视化这些天的数据
        visualize_zero_volatility_days(X, zero_vol_details, observations_per_day, dataset_name)

    else:
        print("✓ 没有发现波动率为0的日期\n")

    return zero_vol_details


def visualize_zero_volatility_days(X, zero_vol_details, observations_per_day, dataset_name):
    """
    可视化波动率为0的日期的数据
    """
    n_zero_days = len(zero_vol_details)

    if n_zero_days == 0:
        return

    # 最多显示前12天
    n_plot = min(n_zero_days, 12)

    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    fig.suptitle(f'Zero Volatility Days - {dataset_name}', fontsize=16, fontweight='bold')

    for i in range(n_plot):
        detail = zero_vol_details[i]
        start_idx = detail['start_idx']
        end_idx = detail['end_idx']
        daily_data = X[start_idx:end_idx]

        axes[i].plot(daily_data, marker='o', markersize=4, linewidth=1)
        axes[i].set_title(f"Day {detail['day_index']}\n{detail['date']}", fontsize=10)
        axes[i].set_xlabel('Hour' if observations_per_day == 24 else 'Time Point')
        axes[i].set_ylabel('OT Value')
        axes[i].grid(True, alpha=0.3)

        # 添加统计信息
        info_text = f"σ={detail['volatility']:.2e}\n"
        info_text += f"μ={detail['mean']:.2f}\n"
        info_text += f"unique={detail['unique_values']}"
        axes[i].text(0.02, 0.98, info_text,
                    transform=axes[i].transAxes,
                    verticalalignment='top',
                    fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 隐藏多余的子图
    for i in range(n_plot, 12):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'{dataset_name}_zero_volatility_days.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"可视化已保存到: {dataset_name}_zero_volatility_days.png\n")


def check_data_quality(data_path, dataset_name, observations_per_day):
    """
    检查数据质量问题
    """
    print(f"\n{'='*60}")
    print(f"数据质量检查 - {dataset_name}")
    print(f"{'='*60}\n")

    df = pd.read_csv(data_path)
    X = df['OT'].values

    # 检查重复值
    n_days = len(X) // observations_per_day
    consecutive_same = []

    for i in range(len(X) - 1):
        if X[i] == X[i+1]:
            consecutive_same.append(i)

    print(f"连续相同值的数据点数量: {len(consecutive_same)}")
    print(f"占比: {len(consecutive_same)/len(X)*100:.2f}%\n")

    # 检查每天的数据变化
    low_variance_days = []

    for d in range(n_days):
        start_idx = d * observations_per_day
        end_idx = start_idx + observations_per_day
        daily_data = X[start_idx:end_idx]

        variance = np.var(daily_data, ddof=1)
        if variance < 0.01:  # 方差小于0.01
            low_variance_days.append((d, variance))

    print(f"低方差天数 (方差 < 0.01): {len(low_variance_days)}")
    if len(low_variance_days) > 0:
        print("前10个低方差天:")
        for d, var in low_variance_days[:10]:
            print(f"  Day {d}: 方差 = {var:.6f}, 标准差 = {np.sqrt(var):.6f}")
    print()


if __name__ == "__main__":

    # ETTh1 分析
    print("\n" + "="*60)
    print("ETTh1 数据集分析")
    print("="*60)

    zero_days_h1 = find_zero_volatility_days(
        data_path="COTN/ETT-small/ETTh1.csv",
        dataset_name="ETTh1",
        observations_per_day=24
    )

    check_data_quality(
        data_path="COTN/ETT-small/ETTh1.csv",
        dataset_name="ETTh1",
        observations_per_day=24
    )

    # ETTm1 分析
    print("\n" + "="*60)
    print("ETTm1 数据集分析")
    print("="*60)

    zero_days_m1 = find_zero_volatility_days(
        data_path="COTN/ETT-small/ETTm1.csv",
        dataset_name="ETTm1",
        observations_per_day=96
    )

    check_data_quality(
        data_path="COTN/ETT-small/ETTm1.csv",
        dataset_name="ETTm1",
        observations_per_day=96
    )

    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)
