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


def plot_turbulence_events_grid(dataset_name, csv_path, output_filename):
    """
    绘制单个数据集的12个Turbulence事件，排列成4行3列

    Args:
        dataset_name: 数据集名称
        csv_path: CSV文件路径
        output_filename: 输出文件名
    """
    # 读取数据
    df = pd.read_csv(csv_path)

    # 找到事件
    events, threshold_90 = find_turbulence_events(df, n_events=12)

    print(f"\n{dataset_name}:")
    print(f"  90%分位数: {threshold_90:.4f}")
    print(f"  找到 {len(events)} 个事件")

    # 创建4行3列的子图
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle(f'{dataset_name} - Turbulence Events (Rising Trends Breaking 90th Percentile)',
                 fontsize=16, fontweight='bold', y=0.995)

    # 蓝色和橙色配色
    line_color = '#1f77b4'  # 蓝色
    threshold_color = '#ff7f0e'  # 橙色

    for i, event in enumerate(events):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        x = np.arange(len(event['data']))

        # 绘制Turbulence曲线（蓝色）
        ax.plot(x, event['data'],
               linewidth=2.5, alpha=0.9,
               color=line_color,
               marker='o', markersize=5,
               markerfacecolor=line_color,
               markeredgecolor='white',
               markeredgewidth=0.5)

        # 添加90%分位线（橙色）
        ax.axhline(y=threshold_90, color=threshold_color, linestyle='--',
                  linewidth=2, alpha=0.8,
                  label=f'90th Percentile')

        # 填充曲线与阈值线之间的区域（橙色）
        # 只填充曲线高于阈值线的部分
        y_fill = np.maximum(event['data'], threshold_90)
        ax.fill_between(x, threshold_90, y_fill,
                        alpha=0.3, color=threshold_color)

        # 设置标签
        if row == 3:  # 最后一行
            ax.set_xlabel('Time Steps', fontsize=10)
        if col == 0:  # 第一列
            ax.set_ylabel('Turbulence Score', fontsize=10)

        # 网格
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

        # 设置y轴范围
        ax.set_ylim(bottom=-0.02, top=max(event['data'].max() * 1.1, threshold_90 * 1.2))

        # 只在第一个子图显示图例
        if i == 0:
            ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

        # 设置刻度字体大小
        ax.tick_params(axis='both', which='major', labelsize=9)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"  图表已保存到: {output_filename}")

    # 打印事件详情
    print(f"\n  事件详情:")
    for i, event in enumerate(events):
        print(f"    Event {i+1}: Day {event['peak_idx']}, "
              f"Peak={event['peak_value']:.4f}, "
              f"Slope={event['slope']:.4f}")


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

    # 检查文件是否存在并处理每个数据集
    import os
    for name, path in datasets:
        if os.path.exists(path):
            print(f"\n处理数据集: {name}")
            output_file = f"{name}_turbulence_events_grid.png"
            plot_turbulence_events_grid(name, path, output_file)
        else:
            print(f"\n✗ 未找到数据集: {name} ({path})")

    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)
