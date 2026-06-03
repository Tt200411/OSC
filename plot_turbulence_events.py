import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def find_turbulence_events(df, n_events=12, window_size=10):
    """
    找出Turbulence呈现上升趋势并突破90%分位的事件

    Args:
        df: DataFrame with TurbulenceScore column
        n_events: 需要找到的事件数量
        window_size: 事件窗口大小（前后各多少天）

    Returns:
        events: 事件列表，每个事件包含索引和数据
    """
    turbulence = df['TurbulenceScore'].values

    # 计算90%分位数
    valid_turb = turbulence[~np.isnan(turbulence)]
    threshold_90 = np.percentile(valid_turb, 90)

    # 找到所有突破90%分位的点
    breakthrough_indices = np.where(turbulence > threshold_90)[0]

    events = []
    used_indices = set()

    for idx in breakthrough_indices:
        # 避免重复选择相近的事件
        if idx in used_indices:
            continue

        # 确保有足够的前后数据
        if idx < window_size or idx >= len(turbulence) - 5:
            continue

        # 提取事件窗口
        start_idx = idx - window_size
        end_idx = idx + 5

        event_data = turbulence[start_idx:end_idx]

        # 检查是否有NaN
        if np.any(np.isnan(event_data)):
            continue

        # 检查是否呈现上升趋势
        # 计算前半部分的趋势
        pre_event = event_data[:window_size]

        # 使用线性回归检查趋势
        x = np.arange(len(pre_event))
        slope = np.polyfit(x, pre_event, 1)[0]

        # 只选择上升趋势的事件（斜率为正）
        if slope > 0:
            events.append({
                'peak_idx': idx,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'data': event_data,
                'slope': slope,
                'peak_value': turbulence[idx]
            })

            # 标记已使用的索引范围
            for i in range(start_idx, end_idx):
                used_indices.add(i)

        if len(events) >= n_events:
            break

    # 按峰值排序，选择最显著的事件
    events.sort(key=lambda x: x['peak_value'], reverse=True)

    return events[:n_events], threshold_90


def plot_all_turbulence_events(datasets, output_filename='turbulence_events_all.png'):
    """
    绘制所有数据集的Turbulence事件

    Args:
        datasets: 数据集列表，每个包含 (name, csv_path)
        output_filename: 输出文件名
    """
    n_datasets = len(datasets)
    n_events = 12

    # 创建图表
    fig, axes = plt.subplots(n_datasets, 1, figsize=(20, 5 * n_datasets))

    if n_datasets == 1:
        axes = [axes]

    fig.suptitle('Turbulence Events: Rising Trends Breaking 90th Percentile',
                 fontsize=18, fontweight='bold', y=0.995)

    colors = plt.cm.tab20(np.linspace(0, 1, n_events))

    for dataset_idx, (dataset_name, csv_path) in enumerate(datasets):
        ax = axes[dataset_idx]

        # 读取数据
        df = pd.read_csv(csv_path)

        # 找到事件
        events, threshold_90 = find_turbulence_events(df, n_events=n_events)

        print(f"\n{dataset_name}:")
        print(f"  90%分位数: {threshold_90:.4f}")
        print(f"  找到 {len(events)} 个事件")

        # 绘制每个事件
        for i, event in enumerate(events):
            x = np.arange(len(event['data']))
            ax.plot(x, event['data'],
                   linewidth=2, alpha=0.8,
                   color=colors[i],
                   marker='o', markersize=4,
                   label=f"Event {i+1} (Day {event['peak_idx']})")

            # 标记峰值点
            peak_x = event['peak_idx'] - event['start_idx']
            ax.scatter([peak_x], [event['peak_value']],
                      color=colors[i], s=100,
                      marker='*', zorder=5,
                      edgecolors='black', linewidths=1.5)

        # 添加90%分位线
        ax.axhline(y=threshold_90, color='red', linestyle='--',
                  linewidth=2, alpha=0.7,
                  label=f'90th Percentile ({threshold_90:.3f})')

        # 设置标题和标签
        ax.set_title(f'{dataset_name} - Turbulence Events',
                    fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Time Steps (relative to event)', fontsize=12)
        ax.set_ylabel('Turbulence Score', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

        # 添加图例
        ax.legend(loc='upper left', fontsize=8, ncol=2,
                 framealpha=0.9, edgecolor='black')

        # 设置y轴范围
        ax.set_ylim(bottom=-0.05)

        # 添加统计信息
        stats_text = f"Events: {len(events)}\n"
        stats_text += f"Avg Peak: {np.mean([e['peak_value'] for e in events]):.3f}\n"
        stats_text += f"Avg Slope: {np.mean([e['slope'] for e in events]):.4f}"

        ax.text(0.98, 0.98, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 打印事件详情
        print(f"\n  事件详情:")
        for i, event in enumerate(events):
            print(f"    Event {i+1}: Day {event['peak_idx']}, "
                  f"Peak={event['peak_value']:.4f}, "
                  f"Slope={event['slope']:.4f}")

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\n图表已保存到: {output_filename}")


if __name__ == "__main__":

    # 定义数据集
    datasets = [
        ("ETTh1", "ETTh1_turbulence_complete.csv"),
        ("ETTh2", "ETTh2_turbulence_complete.csv"),
        ("ETTm1", "ETTm1_turbulence_complete.csv"),
        ("ETTm2", "ETTm2_turbulence_complete.csv")
    ]

    print("="*60)
    print("开始提取和可视化Turbulence事件")
    print("="*60)

    # 检查文件是否存在
    import os
    available_datasets = []
    for name, path in datasets:
        if os.path.exists(path):
            available_datasets.append((name, path))
            print(f"✓ 找到数据集: {name}")
        else:
            print(f"✗ 未找到数据集: {name} ({path})")

    if len(available_datasets) == 0:
        print("\n错误: 没有找到任何数据集文件!")
    else:
        print(f"\n共找到 {len(available_datasets)} 个数据集")
        print("="*60)

        # 绘制所有事件
        plot_all_turbulence_events(available_datasets)

        print("\n" + "="*60)
        print("分析完成!")
        print("="*60)
